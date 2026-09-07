#!/usr/bin/env python3
"""Export confirmed annotations from perch-hoplite databases to JSON.

Writes one file per month per class, using bare recording filenames (not local
paths) so the output is portable. Deduplicates on (recording, offset, label):
repeated runs of merge_dbs.py can insert the same annotation more than once, and
those extra rows are artifacts, not independent labels.

Usage
-----
    ./export_labels.py --out labels/ \
        2018_04=/path/to/db/MARS_20180401_20180430_32kHz_norm \
        2018_05=/path/to/db/MARS_20180501_20180531_32kHz_norm

Each positional argument is MONTHKEY=DBDIR, where DBDIR contains hoplite.sqlite.

Output
------
    labels_<monthkey>_<class>.json   one per month per class
    counts.json                      per-month, per-class summary

The schema of each annotation entry is documented in docs/DATA.md.
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import struct
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# label_type in the hoplite schema: 1 = positive, 2 = weak negative
LABEL_TYPE_MAP = {1: "positive", 2: "negative"}

# Every annotation in this release was made by one annotator. The provenance
# column is not used for these four months: early review sessions ran with a
# generic annotator id, and repeated merge_dbs.py runs appended "_merged" to it.
# Newer databases record per-annotator identity correctly. See labels/README.md
# and docs/DATA.md for how labeling responsibility was divided.
ANNOTATOR = "duane"


def filename_to_utc_epoch(filename):
    """MARS_YYYYMMDD_HHMMSS_*.wav -> UTC epoch seconds, or None."""
    m = re.search(r"MARS_(\d{8})_(\d{6})", filename)
    if not m:
        return None
    dt = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
    return int(dt.replace(tzinfo=timezone.utc).timestamp())


def window_start(blob):
    """First value of the offsets blob: the window start in seconds.

    Each annotation covers exactly one 5-second window, stored as a packed
    (start, end) pair of little-endian doubles.
    """
    if not isinstance(blob, (bytes, bytearray)) or len(blob) < 8:
        return 0.0
    return struct.unpack_from("<d", blob)[0]


def export_month(month_key, db_dir, out_dir):
    sqlite_path = db_dir / "hoplite.sqlite"
    if not sqlite_path.exists():
        sys.exit("error: no hoplite.sqlite in %s" % db_dir)

    con = sqlite3.connect(sqlite_path)
    rows = con.execute(
        """
        SELECT r.filename, a.offsets, a.label, a.label_type
        FROM annotations a
        JOIN recordings r ON a.recording_id = r.id
        ORDER BY r.filename, a.offsets, a.label
        """
    ).fetchall()
    con.close()

    by_class = defaultdict(list)
    recordings = defaultdict(set)
    seen = set()
    n_rows = 0
    n_dupes = 0

    for filename, blob, label, label_type in rows:
        n_rows += 1
        start_s = window_start(blob)

        key = (filename, round(start_s, 3), label)
        if key in seen:
            n_dupes += 1
            continue
        seen.add(key)

        by_class[label].append(
            {
                "species": label,
                "recording_32khz": filename,
                "annotation_offset_s": round(start_s, 1),
                "frame_index": int(start_s / 5),
                "recording_start_utc_epoch": filename_to_utc_epoch(filename),
                "label": 1 if label_type == 1 else 0,
                "label_type": LABEL_TYPE_MAP.get(label_type, str(label_type)),
                "annotator": ANNOTATOR,
                "month": month_key,
            }
        )
        recordings[label].add(filename)

    summary = {
        "rows": n_rows,
        "distinct": n_rows - n_dupes,
        "duplicates_dropped": n_dupes,
        "classes": {},
    }

    for label, entries in sorted(by_class.items()):
        safe = label.replace(" ", "_")
        path = out_dir / ("labels_%s_%s.json" % (month_key, safe))
        payload = {
            "month": month_key,
            "species": label,
            "count": len(entries),
            "n_positive": sum(1 for e in entries if e["label"] == 1),
            "n_negative": sum(1 for e in entries if e["label"] == 0),
            "n_recordings": len(recordings[label]),
            "annotations": entries,
        }
        path.write_text(json.dumps(payload, indent=2) + "\n")
        summary["classes"][label] = {
            "count": payload["count"],
            "n_positive": payload["n_positive"],
            "n_negative": payload["n_negative"],
            "n_recordings": payload["n_recordings"],
        }
        print("  %s: %d" % (path.name, len(entries)))

    if n_dupes:
        print("  (%d duplicate rows dropped)" % n_dupes)

    return summary


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=Path, required=True,
                    help="output directory for the JSON files")
    ap.add_argument("months", nargs="+", metavar="MONTHKEY=DBDIR",
                    help="e.g. 2018_05=/path/to/db/MARS_20180501_20180531_32kHz_norm")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    counts = {}
    for spec in args.months:
        if "=" not in spec:
            sys.exit("error: expected MONTHKEY=DBDIR, got %r" % spec)
        month_key, db = spec.split("=", 1)
        print("%s:" % month_key)
        counts[month_key] = export_month(month_key, Path(db), args.out)

    (args.out / "counts.json").write_text(json.dumps(counts, indent=2) + "\n")

    total = sum(m["distinct"] for m in counts.values())
    dupes = sum(m["duplicates_dropped"] for m in counts.values())
    print("\nTotal: %d distinct annotations across %d months" % (total, len(counts)))
    if dupes:
        print("       %d duplicate rows dropped" % dupes)
    print("Wrote %s/counts.json" % args.out)


if __name__ == "__main__":
    main()
