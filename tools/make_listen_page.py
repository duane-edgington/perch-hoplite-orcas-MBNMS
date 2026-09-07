#!/usr/bin/env python3
"""make_listen_page.py — build a static listening page from confirmed labels.

Deliberately uses the same settings as the project's Gradio review interface,
because those settings were arrived at the hard way:

  * mel spectrograms with the viridis colormap (src/spectrogram.py)
  * 5-second clips peak-normalized to -3 dBFS, matching make_audio_loader()
    in phase2_classify.py -- raw MARS audio is far too quiet to hear
  * 30 seconds of surrounding context, centered on the window and peak-
    normalized to 0.5, matching load_30s_context() in src/audio.py

The context is not decoration. Whether a call belongs to one animal's bout,
differs from its surroundings, sits in isolation, or is masked by ship noise
are all questions the 5-second window cannot answer.

A small, curated page: two or three windows per class. Browsing the whole
dataset is what labels/ is for.

Usage
-----
    ./make_listen_page.py \
        --labels labels/ \
        --selection tools/listen_selection.json \
        --audio-root /mnt/PAM_Analysis/GoogleMultiSpeciesWhaleModel2/resampled_32kHz \
        --out docs/listen

Audio is located as <audio-root>/<YYYY>/<MM>/<filename>.

Output
------
    <out>/index.html
    <out>/clips/*.flac      5-second windows and 30-second context
    <out>/spec/*.png
    <out>/manifest.json
"""
from __future__ import annotations

import argparse
import base64
import glob
import html
import json
import re
import sys
import warnings
from collections import defaultdict
from pathlib import Path

WINDOW_S = 5.0
CONTEXT_S = 30.0

# Match the review interface exactly.
SPEC_TYPE = "mel"
COLORMAP = "viridis"
CLIP_PEAK = 10 ** (-3.0 / 20)   # -3 dBFS, per make_audio_loader()
CTX_PEAK = 0.5                  # per load_30s_context()

# Rendered large: these are meant to be read one per row, not thumbnailed.
CLIP_FIGSIZE, CLIP_DPI = (9.0, 4.0), 130
CTX_FIGSIZE, CTX_DPI = (13.0, 4.2), 130

CLASS_ORDER = ["orca_call", "humpback_song", "dolphin_call", "ship_noise", "other"]

CLASS_TITLE = {
    "orca_call": "Killer whale",
    "humpback_song": "Humpback whale",
    "dolphin_call": "Dolphin",
    "ship_noise": "Ship noise",
    "other": "Other",
}

CLASS_BLURB = {
    "orca_call": "Killer whale vocalization. In this archive the animals are "
                 "predominantly Bigg's killer whales, whose calls come in brief "
                 "discrete bursts rather than long sustained bouts.",
    "humpback_song": "Humpback vocalization. The class name is broader than it "
                     "sounds: it covers social calls as well as song, which is "
                     "part of why humpback is the hardest class to learn.",
    "dolphin_call": "Delphinid vocalization other than killer whale — clicks, "
                    "whistles and buzzes, usually higher and faster than orca calls.",
    "ship_noise": "Vessel noise. Broadband and mechanical, with none of the tonal "
                  "structure biological calls show. In this dataset it correlates "
                  "positively with orca presence: vessels arrive once orcas are seen.",
    "other": "A real sound that fits none of the named classes. Not the same as "
             "silence: quiet background is a separate label that never appears as "
             "a detection.",
}


def audio_path(root: Path, filename: str) -> Path:
    m = re.match(r"MARS_(\d{4})(\d{2})\d{2}_", filename)
    if not m:
        sys.exit("error: cannot parse year/month from %r" % filename)
    return root / m.group(1) / m.group(2) / filename


def normalize(audio, target_peak: float):
    """Peak-normalize for playback. Raw MARS audio is inaudible without this."""
    import numpy as np

    peak = float(np.abs(audio).max())
    if peak > 1e-8:
        audio = audio * (target_peak / peak)
    return audio


def read_window(path: Path):
    """The 5-second window, normalized to -3 dBFS."""
    import soundfile as sf

    def _read(start_s, dur_s):
        info = sf.info(str(path))
        sr = info.samplerate
        start_s = max(0.0, min(start_s, max(0.0, info.duration - dur_s)))
        a, _ = sf.read(str(path), start=int(round(start_s * sr)),
                       frames=int(round(dur_s * sr)), dtype="float32",
                       always_2d=False)
        if a.ndim > 1:
            a = a[:, 0]
        return a, sr, start_s, info.duration

    return _read


def context_bounds(offset_s: float, file_dur: float):
    """Centre CONTEXT_S on the window, exactly as load_30s_context() does."""
    centre = offset_s + WINDOW_S / 2.0
    start = max(0.0, centre - CONTEXT_S / 2.0)
    end = min(file_dur, start + CONTEXT_S)
    start = max(0.0, end - CONTEXT_S)
    return start, end


def write_audio(path: Path, audio, sr: int, fmt: str) -> Path:
    import soundfile as sf

    path = path.with_suffix(".flac" if fmt == "flac" else ".wav")
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sr,
             format="FLAC" if fmt == "flac" else "WAV", subtype="PCM_16")
    return path


def write_png(path: Path, data_uri: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(base64.b64decode(data_uri.split(",", 1)[1]))


def slug(filename: str, offset: float) -> str:
    return "%s_%05.0fs" % (filename.replace("_resampled_32kHz.wav", ""), offset)


def utc_label(filename: str, offset: float) -> str:
    from datetime import datetime, timedelta, timezone

    m = re.search(r"MARS_(\d{8})_(\d{6})", filename)
    if not m:
        return ""
    dt = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
    dt = (dt.replace(tzinfo=timezone.utc) + timedelta(seconds=offset))
    return dt.strftime("%-d %B %Y, %H:%M:%S UTC")


def build_card(entry, out: Path, audio_root: Path, spectro, fmt: str,
               made: set, no_context: bool) -> dict:
    filename = entry["file"]
    offset = float(entry["offset"])
    cls = entry.get("class", "")
    name = slug(filename, offset)
    if name in made:
        return {}
    src = audio_path(audio_root, filename)
    if not src.exists():
        print("  MISSING %s" % src)
        return {}

    reader = read_window(src)

    # 5-second window
    audio, sr, _, file_dur = reader(offset, WINDOW_S)
    clip = write_audio(out / "clips" / name, normalize(audio, CLIP_PEAK), sr, fmt)
    write_png(out / "spec" / (name + ".png"),
              spectro(audio, sr, spec_type=SPEC_TYPE, colormap=COLORMAP,
                      figsize=CLIP_FIGSIZE, dpi=CLIP_DPI))
    made.add(name)

    card = {
        "title": entry.get("title") or utc_label(filename, offset),
        "class": cls,
        "file": filename,
        "offset_s": offset,
        "note": entry.get("note", ""),
        "when": utc_label(filename, offset),
        "clip": "clips/%s" % clip.name,
        "png": "spec/%s.png" % name,
    }

    # 30-second context, window marked
    if not no_context:
        cstart, cend = context_bounds(offset, file_dur)
        caudio, csr, cstart, _ = reader(cstart, cend - cstart)
        hl = offset - cstart
        ctx = write_audio(out / "clips" / (name + "_ctx"),
                          normalize(caudio, CTX_PEAK), csr, fmt)
        write_png(out / "spec" / (name + "_ctx.png"),
                  spectro(caudio, csr, spec_type=SPEC_TYPE, colormap=COLORMAP,
                          highlight_start=hl, highlight_end=hl + WINDOW_S,
                          figsize=CTX_FIGSIZE, dpi=CTX_DPI))
        card.update({"ctx_clip": "clips/%s" % ctx.name,
                     "ctx_png": "spec/%s_ctx.png" % name,
                     "ctx_span": "%.0f–%.0f s" % (cstart, cend),
                     "ctx_seconds": round(cend - cstart, 1)})

    print("  %-34s %s" % (name, "clip + context" if not no_context else "clip"))
    return card


def sample_by_day(entries, n):
    """Pick up to n entries spread across days, so a gallery is not one afternoon."""
    by_day = defaultdict(list)
    for a in entries:
        by_day[a["recording_32khz"][5:13]].append(a)
    for d in by_day:
        by_day[d].sort(key=lambda a: (a["recording_32khz"], a["annotation_offset_s"]))
    picked, days, i = [], sorted(by_day), 0
    while len(picked) < n and any(by_day[d] for d in days):
        d = days[i % len(days)]
        if by_day[d]:
            picked.append(by_day[d].pop(0))
        i += 1
    return picked


def load_labels(labels_dir: Path):
    by_class = defaultdict(list)
    for f in sorted(glob.glob(str(labels_dir / "labels_*.json"))):
        for a in json.load(open(f))["annotations"]:
            if a["label"] == 1:
                by_class[a["species"]].append(a)
    return by_class


CSS = """
:root { color-scheme: dark; }
* { box-sizing: border-box; }
body { margin:0; background:#0b1020; color:#e2e8f0;
  font:16px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif; }
.wrap { max-width:1320px; margin:0 auto; padding:2.5rem 1.25rem 4rem; }
h1 { font-size:2rem; margin:0 0 .4rem; letter-spacing:-.02em; }
h2 { font-size:1.45rem; margin:3.5rem 0 .6rem; border-bottom:1px solid #22304d;
  padding-bottom:.45rem; }
h2 .count { font-weight:400; color:#7d8ca6; font-size:.8rem; }
.sub { color:#93a3bb; margin:0 0 1.4rem; }
.lead { color:#c3cddd; max-width:70ch; }
a { color:#7dd3fc; }
nav { margin:1.6rem 0 0; display:flex; flex-wrap:wrap; gap:.5rem; }
nav a { background:#111827; border:1px solid #22304d; border-radius:999px;
  padding:.3rem .85rem; font-size:.88rem; text-decoration:none; }
nav a:hover { border-color:#7dd3fc; }
.card { background:#111827; border:1px solid #22304d; border-radius:12px;
  padding:1.15rem; margin:1.4rem 0; }
.card h3 { font-size:1.05rem; margin:0 0 .15rem; font-weight:600; }
.card .note { font-size:.92rem; color:#b6c2d4; margin:.35rem 0 .9rem; max-width:78ch; }
.panel { margin-top:1.1rem; }
.panel:first-of-type { margin-top:0; }
.lbl { font-size:.82rem; color:#93a3bb; margin-bottom:.35rem; }
.lbl b { color:#cbd5e1; font-weight:600; }
.card img { width:100%; height:auto; border-radius:8px; display:block; }
audio { width:100%; height:36px; margin-top:.45rem; }
.meta { font-size:.75rem; color:#7d8ca6; margin-top:.9rem;
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace; word-break:break-all; }
footer { margin-top:4rem; padding-top:1.3rem; border-top:1px solid #22304d;
  font-size:.87rem; color:#7d8ca6; max-width:78ch; }
"""


def card_html(c):
    b = ['<div class="card">', "<h3>%s</h3>" % html.escape(c["title"])]
    if c.get("note"):
        b.append('<p class="note">%s</p>' % html.escape(c["note"]))
    b.append('<div class="panel"><div class="lbl"><b>The 5-second window</b> '
             '&mdash; what the classifier scores</div>')
    b.append('<img src="%s" alt="Spectrogram of the 5-second window">' % c["png"])
    b.append('<audio controls preload="none" src="%s"></audio></div>' % c["clip"])
    if c.get("ctx_png"):
        b.append('<div class="panel"><div class="lbl"><b>%g seconds of context</b> '
                 '&mdash; the window is marked; %s within the recording</div>'
                 % (c.get("ctx_seconds", 30), html.escape(c["ctx_span"])))
        b.append('<img src="%s" alt="Spectrogram of the surrounding context">'
                 % c["ctx_png"])
        b.append('<audio controls preload="none" src="%s"></audio></div>' % c["ctx_clip"])
    b.append('<div class="meta">%s &nbsp;+%.1f s</div>'
             % (html.escape(c["file"]), c["offset_s"]))
    b.append("</div>")
    return "\n".join(b)


def render_html(galleries, totals, meta):
    p = ["<!DOCTYPE html>", '<html lang="en"><head><meta charset="utf-8">',
         '<meta name="viewport" content="width=device-width,initial-scale=1">',
         "<title>%s</title>" % html.escape(meta["title"]),
         "<style>%s</style></head><body><div class=\"wrap\">" % CSS,
         "<h1>%s</h1>" % html.escape(meta["title"]),
         '<p class="sub">%s</p>' % meta["subtitle"],
         '<p class="lead">%s</p>' % meta["lead"], "<nav>"]
    for cls in CLASS_ORDER:
        if galleries.get(cls):
            p.append('<a href="#%s">%s</a>' % (cls.replace("_", "-"),
                                               html.escape(CLASS_TITLE.get(cls, cls))))
    p.append("</nav>")
    for cls in CLASS_ORDER:
        cards = galleries.get(cls)
        if not cards:
            continue
        p.append('<h2 id="%s">%s <span class="count">&mdash; %d of %d confirmed '
                 'windows</span></h2>'
                 % (cls.replace("_", "-"), html.escape(CLASS_TITLE.get(cls, cls)),
                    len(cards), totals.get(cls, len(cards))))
        if CLASS_BLURB.get(cls):
            p.append('<p class="lead">%s</p>' % html.escape(CLASS_BLURB[cls]))
        p += [card_html(c) for c in cards]
    p.append("<footer>%s</footer>" % meta["footer"])
    p.append("</div></body></html>")
    return "\n".join(p)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--audio-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--selection", type=Path, default=None,
                    help="JSON with hand-picked windows per class and page text")
    ap.add_argument("--per-class", type=int, default=3,
                    help="cards per class when topping up from labels (default 3)")
    ap.add_argument("--orca-per-class", type=int, default=6,
                    help="cards for orca_call specifically (default 6)")
    ap.add_argument("--audio-format", choices=("flac", "wav"), default="flac")
    ap.add_argument("--no-context", action="store_true",
                    help="skip the 30-second context panels")
    args = ap.parse_args()

    mod_dir = Path(__file__).resolve().parent.parent / "pipeline" / "src"
    sys.path.insert(0, str(mod_dir))
    try:
        from spectrogram import make_spectrogram_image as spectro
    except ImportError as exc:
        sys.exit("error: could not import spectrogram.py from %s (%s)" % (mod_dir, exc))

    import inspect
    if "figsize" not in inspect.signature(spectro).parameters:
        sys.exit("error: spectrogram.py is missing the figsize/dpi parameters.\n"
                 "       Update pipeline/src/spectrogram.py before running this.")

    warnings.filterwarnings("ignore", message="Empty filters detected")

    sel = json.loads(args.selection.read_text()) if args.selection else {}
    meta = sel.get("page", {})
    meta.setdefault("title", "Listen")
    for k in ("subtitle", "lead", "footer"):
        meta.setdefault(k, "")

    picks_by_class = defaultdict(list)
    for e in sel.get("cards", []):
        picks_by_class[e["class"]].append(e)

    by_class = load_labels(args.labels)
    totals = {c: len(v) for c, v in by_class.items()}

    args.out.mkdir(parents=True, exist_ok=True)
    made: set = set()
    galleries = {}

    for cls in CLASS_ORDER:
        want = args.orca_per_class if cls == "orca_call" else args.per_class
        entries = list(picks_by_class.get(cls, []))
        if len(entries) < want and cls in by_class:
            chosen = {(e["file"], float(e["offset"])) for e in entries}
            for a in sample_by_day(by_class[cls], want * 4 + 8):
                key = (a["recording_32khz"], a["annotation_offset_s"])
                if key in chosen:
                    continue
                entries.append({"file": a["recording_32khz"],
                                "offset": a["annotation_offset_s"], "class": cls})
                if len(entries) >= want:
                    break
        if not entries:
            continue
        print("%s:" % cls)
        cards = []
        for e in entries:
            if len(cards) >= want:
                break
            c = build_card(e, args.out, args.audio_root, spectro,
                           args.audio_format, made, args.no_context)
            if c:
                cards.append(c)
        galleries[cls] = cards

    (args.out / "index.html").write_text(render_html(galleries, totals, meta))
    (args.out / "manifest.json").write_text(
        json.dumps({"galleries": galleries, "totals": totals,
                    "settings": {"spec_type": SPEC_TYPE, "colormap": COLORMAP,
                                 "clip_peak_dbfs": -3.0, "context_peak": CTX_PEAK,
                                 "context_s": CONTEXT_S}}, indent=2) + "\n")
    (args.out.parent / ".nojekyll").touch()

    n = sum(len(v) for v in galleries.values())
    size = sum(p.stat().st_size for p in args.out.rglob("*") if p.is_file())
    print("\n%d cards -> %s  (%.1f MB)" % (n, args.out / "index.html", size / 1024**2))


if __name__ == "__main__":
    main()
