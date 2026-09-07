#!/usr/bin/env python3
"""make_listen_page.py — build a static listening page from confirmed labels.

Produces a self-contained page with two parts:

  1. One clear example of each class, hand-picked.
  2. A browsable gallery per class, sampled automatically from the label files
     and spread across days so it shows variety rather than one busy recording.

Every card is a 5-second window with a spectrogram and an audio player. Output
is plain static files, ready to serve from GitHub Pages with no build step.

Spectrograms come from pipeline/src/spectrogram.py, so they match what the
project's annotation interface shows.

Usage
-----
    ./make_listen_page.py \
        --labels labels/ \
        --exemplars tools/listen_selection.json \
        --audio-root /mnt/PAM_Analysis/GoogleMultiSpeciesWhaleModel2/resampled_32kHz \
        --out docs/listen \
        --per-class 12

Audio is located as <audio-root>/<YYYY>/<MM>/<filename>, with year and month
taken from the MARS_<YYYYMMDD>_... filename.

Size
----
Each card costs roughly 160 KB of FLAC plus 145 KB of PNG, so about 300 KB.
--per-class 12 across five classes is roughly 20 MB. Raise or lower to taste;
the full set of 1,351 confirmed windows would be about 400 MB, which is why
this samples instead. Every offset is in labels/, so any window not shown here
can be regenerated from the public audio.

Output
------
    <out>/index.html
    <out>/clips/*.flac      (or .wav with --audio-format wav)
    <out>/spec/*.png
    <out>/manifest.json
"""
from __future__ import annotations

import argparse
import base64
import glob
import html
import json
import os
import re
import sys
import warnings
from collections import defaultdict
from pathlib import Path

WINDOW_S = 5.0

CLASS_ORDER = ["orca_call", "humpback_song", "dolphin_call", "ship_noise", "other"]

CLASS_TITLE = {
    "orca_call": "Killer whale",
    "humpback_song": "Humpback whale",
    "dolphin_call": "Dolphin",
    "ship_noise": "Ship noise",
    "other": "Other",
}

# Per the spectrogram module: linear STFT suits orca and dolphin, mel suits humpback.
DEFAULT_SPEC = {
    "orca_call": "linear",
    "dolphin_call": "linear",
    "ship_noise": "linear",
    "humpback_song": "mel",
    "other": "mel",
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
    "ship_noise": "Vessel noise. Broadband and mechanical, with no tonal structure "
                  "of the kind biological calls show.",
    "other": "A real sound that fits none of the named classes. Note this is not "
             "the same as silence: quiet background is a separate label that never "
             "appears as a detection.",
}


def audio_path(root: Path, filename: str) -> Path:
    m = re.match(r"MARS_(\d{4})(\d{2})\d{2}_", filename)
    if not m:
        sys.exit("error: cannot parse year/month from %r" % filename)
    return root / m.group(1) / m.group(2) / filename


def read_segment(path: Path, start_s: float, dur_s: float):
    import soundfile as sf

    info = sf.info(str(path))
    sr = info.samplerate
    start_s = max(0.0, min(start_s, max(0.0, info.duration - dur_s)))
    audio, sr = sf.read(str(path), start=int(round(start_s * sr)),
                        frames=int(round(dur_s * sr)),
                        dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio[:, 0]
    return audio, sr, start_s


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
    """'12 May 2018, 08:05 UTC' from the filename stamp plus the offset."""
    from datetime import datetime, timedelta, timezone

    m = re.search(r"MARS_(\d{8})_(\d{6})", filename)
    if not m:
        return ""
    dt = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
    dt = dt.replace(tzinfo=timezone.utc) + timedelta(seconds=offset)
    return dt.strftime("%-d %B %Y, %H:%M UTC")


def build_card(entry, out: Path, audio_root: Path, spectro, fmt: str,
               made: set) -> dict:
    filename = entry["file"]
    offset = float(entry["offset"])
    cls = entry.get("class", "")
    spec = entry.get("spec") or DEFAULT_SPEC.get(cls, "linear")
    name = slug(filename, offset)
    if name in made:
        return {}

    src = audio_path(audio_root, filename)
    if not src.exists():
        print("  MISSING %s" % src)
        return {}

    audio, sr, _ = read_segment(src, offset, WINDOW_S)
    clip = write_audio(out / "clips" / name, audio, sr, fmt)
    write_png(out / "spec" / (name + ".png"), spectro(audio, sr, spec_type=spec))
    made.add(name)

    card = {
        "title": entry.get("title") or utc_label(filename, offset),
        "class": cls,
        "file": filename,
        "offset_s": offset,
        "note": entry.get("note", ""),
        "spec_type": spec,
        "when": utc_label(filename, offset),
        "clip": "clips/%s" % clip.name,
        "png": "spec/%s.png" % name,
    }
    print("  %-34s %s" % (name, spec))
    return card


def sample_by_day(entries, n):
    """Pick up to n entries spread evenly across the days available.

    Round-robins over days, so a class whose windows cluster in one recording
    still yields a gallery that shows more than one afternoon.
    """
    by_day = defaultdict(list)
    for a in entries:
        by_day[a["recording_32khz"][5:13]].append(a)
    for day in by_day:
        by_day[day].sort(key=lambda a: (a["recording_32khz"], a["annotation_offset_s"]))
    picked, days = [], sorted(by_day)
    i = 0
    while len(picked) < n and any(by_day[d] for d in days):
        d = days[i % len(days)]
        if by_day[d]:
            picked.append(by_day[d].pop(0))
        i += 1
    return picked


def load_labels(labels_dir: Path):
    """Return {class: [annotation, ...]} of positives only, plus per-class totals."""
    by_class = defaultdict(list)
    for f in sorted(glob.glob(str(labels_dir / "labels_*.json"))):
        j = json.load(open(f))
        for a in j["annotations"]:
            if a["label"] == 1:
                by_class[a["species"]].append(a)
    return by_class


CSS = """
:root { color-scheme: dark; }
* { box-sizing: border-box; }
body { margin:0; background:#0b1020; color:#e2e8f0;
  font:16px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif; }
.wrap { max-width:1180px; margin:0 auto; padding:2.5rem 1.25rem 4rem; }
h1 { font-size:1.9rem; margin:0 0 .4rem; letter-spacing:-.02em; }
h2 { font-size:1.3rem; margin:3rem 0 .5rem; border-bottom:1px solid #22304d; padding-bottom:.4rem; }
h2 .count { font-weight:400; color:#7d8ca6; font-size:.85rem; }
.sub { color:#93a3bb; margin:0 0 1.4rem; }
.lead { color:#c3cddd; max-width:66ch; }
a { color:#7dd3fc; }
nav { margin:1.6rem 0 0; display:flex; flex-wrap:wrap; gap:.5rem; }
nav a { background:#111827; border:1px solid #22304d; border-radius:999px;
  padding:.28rem .8rem; font-size:.85rem; text-decoration:none; }
nav a:hover { border-color:#7dd3fc; }
.grid { display:grid; gap:1.1rem; grid-template-columns:repeat(auto-fill,minmax(320px,1fr)); }
.card { background:#111827; border:1px solid #22304d; border-radius:10px;
  padding:.85rem; display:flex; flex-direction:column; gap:.5rem; }
.card h3 { font-size:.95rem; margin:0; font-weight:600; }
.card img { width:100%; height:auto; border-radius:6px; display:block; }
.meta { font-size:.72rem; color:#7d8ca6;
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace; word-break:break-all; }
.note { font-size:.85rem; color:#b6c2d4; margin:0; }
audio { width:100%; height:34px; }
.tag { display:inline-block; font-size:.66rem; text-transform:uppercase;
  letter-spacing:.06em; padding:.13rem .45rem; border-radius:4px;
  background:#1e293b; color:#94a3b8; }
footer { margin-top:3.5rem; padding-top:1.25rem; border-top:1px solid #22304d;
  font-size:.85rem; color:#7d8ca6; }
"""


def card_html(c, show_class=True):
    b = ['<div class="card">', "<h3>%s</h3>" % html.escape(c["title"])]
    if show_class and c.get("class"):
        b.append('<div><span class="tag">%s</span></div>' % html.escape(c["class"]))
    b.append('<img src="%s" alt="Spectrogram">' % c["png"])
    b.append('<audio controls preload="none" src="%s"></audio>' % c["clip"])
    if c.get("note"):
        b.append('<p class="note">%s</p>' % html.escape(c["note"]))
    b.append('<div class="meta">%s &nbsp;+%.0f s</div>'
             % (html.escape(c["file"]), c["offset_s"]))
    b.append("</div>")
    return "\n".join(b)


def render_html(exemplars, galleries, totals, meta):
    p = ["<!DOCTYPE html>", '<html lang="en"><head><meta charset="utf-8">',
         '<meta name="viewport" content="width=device-width,initial-scale=1">',
         "<title>%s</title>" % html.escape(meta["title"]),
         "<style>%s</style></head><body><div class=\"wrap\">" % CSS,
         "<h1>%s</h1>" % html.escape(meta["title"]),
         '<p class="sub">%s</p>' % meta["subtitle"],
         '<p class="lead">%s</p>' % meta["lead"]]

    p.append("<nav>")
    if exemplars:
        p.append('<a href="#one-of-each">One of each</a>')
    for cls in CLASS_ORDER:
        if galleries.get(cls):
            p.append('<a href="#%s">%s</a>' % (cls.replace("_", "-"),
                                               html.escape(CLASS_TITLE.get(cls, cls))))
    p.append("</nav>")

    if exemplars:
        p.append('<h2 id="one-of-each">One of each</h2>')
        p.append('<p class="lead">%s</p>' % meta.get("exemplars_lead", ""))
        p.append('<div class="grid">')
        p += [card_html(c) for c in exemplars]
        p.append("</div>")

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
        p.append('<div class="grid">')
        p += [card_html(c, show_class=False) for c in cards]
        p.append("</div>")

    p.append("<footer>%s</footer>" % meta["footer"])
    p.append("</div></body></html>")
    return "\n".join(p)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", type=Path, required=True,
                    help="directory of labels_*.json")
    ap.add_argument("--audio-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--exemplars", type=Path, default=None,
                    help="JSON with hand-picked 'One of each' cards and page text")
    ap.add_argument("--per-class", type=int, default=12,
                    help="gallery cards per class (default 12; ~300 KB each)")
    ap.add_argument("--audio-format", choices=("flac", "wav"), default="flac",
                    help="flac is lossless and about half the size (default)")
    args = ap.parse_args()

    mod_dir = Path(__file__).resolve().parent.parent / "pipeline" / "src"
    sys.path.insert(0, str(mod_dir))
    try:
        from spectrogram import make_spectrogram_image as spectro
    except ImportError as exc:
        sys.exit("error: could not import spectrogram.py from %s (%s)" % (mod_dir, exc))

    warnings.filterwarnings("ignore", message="Empty filters detected")

    sel = json.loads(args.exemplars.read_text()) if args.exemplars else {}
    meta = sel.get("page", {})
    meta.setdefault("title", "Listen")
    for k in ("subtitle", "lead", "footer", "exemplars_lead"):
        meta.setdefault(k, "")

    args.out.mkdir(parents=True, exist_ok=True)
    made: set = set()

    print("one of each:")
    exemplars = [c for c in (build_card(e, args.out, args.audio_root, spectro,
                                        args.audio_format, made)
                             for e in sel.get("exemplars", [])) if c]

    by_class = load_labels(args.labels)
    totals = {c: len(v) for c, v in by_class.items()}
    galleries = {}
    for cls in CLASS_ORDER:
        if cls not in by_class:
            continue
        print("%s:" % cls)
        # Over-sample, then fill to the requested count: some picks collide with
        # the hand-chosen exemplars, and a skipped card should be replaced rather
        # than leave the gallery short.
        picks = sample_by_day(by_class[cls], args.per_class * 3 + 6)
        cards = []
        for a in picks:
            if len(cards) >= args.per_class:
                break
            c = build_card(
                {"file": a["recording_32khz"], "offset": a["annotation_offset_s"],
                 "class": cls},
                args.out, args.audio_root, spectro, args.audio_format, made)
            if c:
                cards.append(c)
        galleries[cls] = cards
        if len(cards) < args.per_class:
            print("  (only %d available)" % len(cards))

    (args.out / "index.html").write_text(
        render_html(exemplars, galleries, totals, meta))
    (args.out / "manifest.json").write_text(json.dumps(
        {"exemplars": exemplars, "galleries": galleries, "totals": totals},
        indent=2) + "\n")
    (args.out.parent / ".nojekyll").touch()

    n = len(exemplars) + sum(len(v) for v in galleries.values())
    size = sum(p.stat().st_size for p in args.out.rglob("*") if p.is_file())
    print("\n%d cards -> %s  (%.1f MB)" % (n, args.out / "index.html", size / 1024**2))


if __name__ == "__main__":
    main()
