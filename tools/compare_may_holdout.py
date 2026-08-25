#!/usr/bin/env python3
"""
compare_may_holdout.py — v4 vs orca_v10 on the May 2018 held-out test month.

May 2018 is the permanent held-out validation month (never trained on by either
v4 or orca_v10, per CLAUDE.md finding #18). This scores both models on the
confirmed-orca ground truth (May 12/13/14/16, reviewed by ear) to answer:
does orca_v10 recognize held-out orca better than v4?

Ground truth = orca_call positive annotations in the May DB.
For each confirmed-orca window, look up the score each model gave it in its
floor-0.0 inference CSV. A window absent from a model's CSV means that model
scored it BELOW 0.0 (a miss at any positive threshold).

Reports, per model: recall at 0.0 / +1.16 / +1.5 / +2.0, plus score
distribution (mean/median/min/max) on the confirmed-orca set. Higher recall
and higher scores on known orca = better on the hold-out.
"""
import csv
import sqlite3
import struct
import sys
from pathlib import Path

DB = "/mnt/PAM_Analysis/perch-hoplite/db/MARS_20180501_20180531_32kHz_norm/hoplite.sqlite"
V4_CSV = "/mnt/PAM_Analysis/perch-hoplite/results/MARS_20180501_20180531_v4_orcaval.csv"
V10_CSV = "/mnt/PAM_Analysis/perch-hoplite/results/MARS_20180501_20180531_v10_orcaval.csv"

THRESHOLDS = [0.0, 1.16, 1.5, 2.0]


def confirmed_orca_keys(db_path):
    """Return set of (filename_stem, round(start,1)) for confirmed orca windows."""
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    rows = cur.execute("""
        SELECT r.filename, a.offsets
        FROM annotations a JOIN recordings r ON r.id = a.recording_id
        WHERE a.label='orca_call' AND a.label_type=1
    """).fetchall()
    con.close()
    keys = {}
    for fname, blob in rows:
        start, _end = struct.unpack('<2d', blob)
        stem = Path(fname).name.replace('.wav', '')
        keys[(stem, round(start, 1))] = None
    return keys


def load_scores(csv_path):
    """Return dict (filename_stem, round(start,1)) -> logit, for orca_call rows."""
    scores = {}
    with open(csv_path) as f:
        r = csv.DictReader(f)
        for row in r:
            if row.get('label') != 'orca_call':
                continue
            stem = Path(row['filename']).name.replace('.wav', '')
            start = round(float(row['window_start']), 1)
            scores[(stem, start)] = float(row['logits'])
    return scores


def summarize(name, gt_keys, model_scores):
    found = {k: model_scores[k] for k in gt_keys if k in model_scores}
    vals = sorted(found.values())
    n_gt = len(gt_keys)
    print(f"\n=== {name} ===")
    print(f"  Confirmed-orca windows: {n_gt}")
    print(f"  Present in model's floor-0.0 output: {len(found)} "
          f"({100*len(found)/n_gt:.1f}%)")
    if vals:
        import statistics
        print(f"  Score on confirmed orca — mean {statistics.mean(vals):.3f}, "
              f"median {statistics.median(vals):.3f}, "
              f"min {vals[0]:.3f}, max {vals[-1]:.3f}")
    print("  Recall at threshold (fraction of confirmed orca scoring >= t):")
    for t in THRESHOLDS:
        n = sum(1 for v in found.values() if v >= t)
        print(f"    >= {t:>4}: {n:4d} / {n_gt}  ({100*n/n_gt:5.1f}%)")
    return found


def main():
    for p in (DB, V4_CSV, V10_CSV):
        if not Path(p).exists():
            print(f"MISSING: {p}"); sys.exit(1)

    gt = confirmed_orca_keys(DB)
    print(f"Ground truth: {len(gt)} confirmed orca_call windows in May 2018 DB")

    v4 = load_scores(V4_CSV)
    v10 = load_scores(V10_CSV)
    print(f"v4 inference rows (orca_call): {len(v4)}")
    print(f"v10 inference rows (orca_call): {len(v10)}")

    f4 = summarize("v4 (production)", gt, v4)
    f10 = summarize("orca_v10 (candidate)", gt, v10)

    # Head-to-head on windows both models scored
    both = set(f4) & set(f10)
    print(f"\n=== Head-to-head (confirmed orca windows scored by BOTH) : {len(both)} ===")
    if both:
        v10_higher = sum(1 for k in both if f10[k] > f4[k])
        v4_higher = sum(1 for k in both if f4[k] > f10[k])
        print(f"  v10 scored higher: {v10_higher}")
        print(f"  v4  scored higher: {v4_higher}")
        mean_delta = sum(f10[k] - f4[k] for k in both) / len(both)
        print(f"  mean(v10 - v4) on shared confirmed orca: {mean_delta:+.3f} "
              f"({'v10 higher' if mean_delta>0 else 'v4 higher'})")

    # Misses: confirmed orca each model dropped below 0.0
    v4_miss = set(gt) - set(f4)
    v10_miss = set(gt) - set(f10)
    print(f"\n=== Misses (confirmed orca scored < 0.0, invisible at any positive threshold) ===")
    print(f"  v4 missed:  {len(v4_miss)}")
    print(f"  v10 missed: {len(v10_miss)}")
    only_v10_saves = v4_miss - v10_miss
    only_v4_saves = v10_miss - v4_miss
    print(f"  caught by v10 but missed by v4: {len(only_v10_saves)}")
    print(f"  caught by v4 but missed by v10: {len(only_v4_saves)}")


if __name__ == "__main__":
    main()
