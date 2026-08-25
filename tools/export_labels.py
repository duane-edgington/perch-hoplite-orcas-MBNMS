#!/usr/bin/env python3
"""Export all human-verified annotations from perch-hoplite DBs to JSON
for the temporal analysis pipeline.
"""
import json, sqlite3, struct, os, re
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict

# ── Config ────────────────────────────────────────────────────────────────
OUT_DIR   = Path("/mnt/PAM_Analysis/perch-hoplite/json_labels")
PAM_AUDIO = Path("/mnt/PAM_Analysis/GoogleMultiSpeciesWhaleModel2/resampled_32kHz")

DATABASES = [
    {
        "db_path":   "/mnt/PAM_Analysis/perch-hoplite/db/MARS_20180401_20180430_32kHz_norm",
        "audio_dir": PAM_AUDIO / "2018/04",
        "month_key": "2018_04",
        "month_label": "April 2018",
    },
    {
        "db_path":   "/mnt/PAM_Analysis/perch-hoplite/db/MARS_20180501_20180531_32kHz_norm",
        "audio_dir": PAM_AUDIO / "2018/05",
        "month_key": "2018_05",
        "month_label": "May 2018",
    },
    {
        "db_path":   "/mnt/PAM_Analysis/perch-hoplite/db/MARS_20201001_20201031_32kHz_norm",
        "audio_dir": PAM_AUDIO / "2020/10",
        "month_key": "2020_10",
        "month_label": "October 2020",
    },
    {
        "db_path":   "/mnt/PAM_Analysis/perch-hoplite/db/MARS_20260401_20260430_32kHz_norm",
        "audio_dir": PAM_AUDIO / "2026/04",
        "month_key": "2026_04",
        "month_label": "April 2026",
    },
]

# label_type: 1=POSITIVE, 2=NEGATIVE/WEAK_NEGATIVE
LABEL_TYPE_MAP = {1: "positive", 2: "negative"}

OUT_DIR.mkdir(parents=True, exist_ok=True)


def filename_to_utc_epoch(filename):
    m = re.search(r'MARS_(\d{8})_(\d{6})', filename)
    if not m:
        return None
    dt = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
    return int(dt.replace(tzinfo=timezone.utc).timestamp())


def export_db(cfg):
    db_path   = cfg["db_path"]
    audio_dir = cfg["audio_dir"]
    month_key = cfg["month_key"]

    con = sqlite3.connect(f"{db_path}/hoplite.sqlite")
    rows = con.execute("""
        SELECT r.filename, a.offsets, a.label, a.label_type, a.provenance
        FROM annotations a
        JOIN recordings r ON a.recording_id = r.id
        ORDER BY r.filename, a.offsets
    """).fetchall()
    con.close()

    # Group into per-species lists
    by_species = defaultdict(list)
    recordings_per_species = defaultdict(set)

    for filename, off_blob, label, label_type, provenance in rows:
        if isinstance(off_blob, (bytes, bytearray)) and len(off_blob) >= 8:
            start_s = struct.unpack_from('<d', off_blob)[0]
        else:
            start_s = 0.0

        epoch = filename_to_utc_epoch(filename)
        wav   = str(audio_dir / filename)

        # Map label_type to label int: positive=1, negative=0
        label_int = 1 if label_type == 1 else 0

        entry = {
            "species":                   label,
            "recording_32khz":           wav,
            "annotation_offset_s":       round(start_s, 1),
            "frame_index":               int(start_s / 5),
            "recording_start_utc_epoch": epoch,
            "label":                     label_int,
            "label_type":                LABEL_TYPE_MAP.get(label_type, str(label_type)),
            "annotator":                 provenance or "analyst",
            "month":                     month_key,
        }
        by_species[label].append(entry)
        recordings_per_species[label].add(filename)

    # Write one JSON per species
    written = {}
    for species, entries in by_species.items():
        safe = species.replace(" ", "_")
        fname = OUT_DIR / f"labels_{month_key}_{safe}.json"
        with open(fname, "w") as f:
            json.dump({
                "month":    month_key,
                "species":  species,
                "count":    len(entries),
                "n_positive": sum(1 for e in entries if e["label"] == 1),
                "n_negative": sum(1 for e in entries if e["label"] == 0),
                "n_recordings": len(recordings_per_species[species]),
                "annotations": entries,
            }, f, indent=2)
        written[species] = {
            "file": str(fname),
            "count": len(entries),
            "n_positive": sum(1 for e in entries if e["label"] == 1),
            "n_negative": sum(1 for e in entries if e["label"] == 0),
            "n_recordings": len(recordings_per_species[species]),
        }
        print(f"  {fname.name}: {len(entries)} annotations")

    return written


# ── Main ──────────────────────────────────────────────────────────────────
inventory = {}

for cfg in DATABASES:
    print(f"\n{cfg['month_label']} ({cfg['month_key']}):")
    written = export_db(cfg)
    inventory[cfg["month_key"]] = {
        "month_label": cfg["month_label"],
        "species": written,
    }

# ── Write inventory.json ─────────────────────────────────────────────────
inv_path = OUT_DIR / "inventory.json"
with open(inv_path, "w") as f:
    json.dump(inventory, f, indent=2)
print(f"\nInventory: {inv_path}")

# ── Write INVENTORY.md ───────────────────────────────────────────────────
lines = ["# Annotation Inventory — perch-hoplite\n",
         "Human-verified annotations exported for temporal analysis pipeline.\n",
         f"Generated: {datetime.now(timezone.utc).isoformat()}\n",
         f"Output directory: {OUT_DIR}\n\n"]

for month_key, mdata in inventory.items():
    lines.append(f"## {mdata['month_label']} (`{month_key}`)\n\n")
    lines.append("| Species | Annotations | Positive | Negative | Recordings |\n")
    lines.append("|---|---|---|---|---|\n")
    for species, sdata in mdata["species"].items():
        lines.append(
            f"| {species} | {sdata['count']} | {sdata['n_positive']} | "
            f"{sdata['n_negative']} | {sdata['n_recordings']} |\n"
        )
    lines.append("\n")

# Add path-to-more section
lines += [
    "## Path to More Annotations\n\n",
    "| Month | DB | Raw orca detections (v4) | Ceiling for annotation |\n",
    "|---|---|---|---|\n",
    "| April 2018 | MARS_20180401_20180430_32kHz_norm | 1,556 | ~200-300 high-confidence |\n",
    "| May 2018 | MARS_20180501_20180531_32kHz_norm | 241 | ~50-100 (May 12 event) |\n",
    "| October 2020 | MARS_20201001_20201031_32kHz_norm | 144 | ~50-80 |\n",
    "| April 2026 | MARS_20260401_20260430_32kHz_norm | 323 | ~50 (likely all humpback FP) |\n\n",
    "May 2018 has **zero annotations** — all inference, no human labels yet.\n",
    "April 2026 has 25 annotations (all humpback FP, labeled as hard negatives for orca).\n\n",
    "## File Naming\n\n",
    "```\n",
    "labels_{YYYY_MM}_{species}.json\n",
    "```\n",
    "One file per species per month. Each file contains all annotations for that\n",
    "species in that month, including negatives (label=0).\n\n",
    "## JSON Schema\n\n",
    "```json\n",
    '{\n',
    '  "species": "humpback_song",\n',
    '  "recording_32khz": "/mnt/PAM_Analysis/.../MARS_YYYYMMDD_HHMMSS_resampled_32kHz.wav",\n',
    '  "annotation_offset_s": 45.0,\n',
    '  "frame_index": 9,\n',
    '  "recording_start_utc_epoch": 1524052752,\n',
    '  "label": 1,\n',
    '  "label_type": "positive",\n',
    '  "annotator": "gradio_gui:analyst",\n',
    '  "month": "2018_04"\n',
    '}\n',
    "```\n",
    "- `label`: 1=positive, 0=negative/weak-negative\n",
    "- `frame_index`: annotation_offset_s / 5 — aligns to Perch .npz and logit CSV frames\n",
    "- `recording_start_utc_epoch`: filename YYYYMMDD_HHMMSS parsed as UTC\n",
]

md_path = OUT_DIR / "INVENTORY.md"
with open(md_path, "w") as f:
    f.writelines(lines)
print(f"Markdown: {md_path}")
print("\nDone.")
