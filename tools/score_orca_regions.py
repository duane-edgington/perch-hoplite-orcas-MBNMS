#!/usr/bin/env python3
"""
score_orca_regions.py — Tabulate orca detections against known ground-truth regions.

Reads the per-month inference CSVs produced by `phase2_classify.py infer`
(columns: idx, project, filename, window_start, window_end, label, logits — one row
per window, argmax label, `logits` = winning score), keeps the orca_call rows, and
sweeps a threshold ladder IN MEMORY. For each threshold it counts orca detections in
each known region and interprets them:

  PRESENT region (orca confirmed) -> detections should be RETAINED as threshold rises.
                                     Reported vs a reference count (retention %).
  ABSENT  region (orca confirmed silent) -> every orca detection is a FALSE POSITIVE.
                                     Should collapse toward 0 as threshold rises.

This directly answers: which threshold keeps the confirmed Bigg's events (Apr 13 / May 12
2018) while killing the April 2026 humpback false positives and holding Oct 2020 at ~0?

Ground truth from CLAUDE.md (edit REGIONS below as annotations firm up):
  Apr 13 2018   — confirmed Bigg's hunting event         (present, ~289 ref)
  May 12 2018   — confirmed event                        (present, ~190 ref)
  May 14 2018   — secondary, probable                    (present, ~45 ref)
  Oct 2020      — confirmed SILENT (Bigg's silent hunt)  (absent, 0)
  Apr 2026      — confirmed no orca vocals (humpback FP) (absent, 0)

Dates are parsed from `filename` (first YYYYMMDD run by default; override --date-regex).

USAGE
  python3 tools/score_orca_regions.py \
      --results-dir /mnt/PAM_Analysis/perch-hoplite/results \
      --pattern '*_v4_orcaval.csv' \
      --thresholds 0.0,1.0,1.16,1.5,2.0 \
      --by-day \
      --out-summary /mnt/PAM_Analysis/perch-hoplite/results/orca_region_scores_v4.csv

  # self-test on synthetic data (no CSVs needed):
  python3 tools/score_orca_regions.py --selftest
"""

import argparse
import csv
import glob
import os
import re
import sys
from collections import defaultdict
from datetime import date, datetime

ORCA_LABEL_DEFAULT = "orca_call"
DATE_REGEX_DEFAULT = r"(20\d{2})(\d{2})(\d{2})"  # YYYYMMDD anywhere in filename

# ── Known ground-truth regions. Edit as expert review firms up. ──────────────────────
# truth: "present" (orca confirmed) or "absent" (orca confirmed silent -> detections=FP)
# ref  : reference detection count for a PRESENT region (retention %); 0 for ABSENT.
REGIONS = [
    {"name": "Apr 13 2018 — Bigg's event (CONFIRMED)",        "start": "2018-04-13", "end": "2018-04-13", "truth": "present", "ref": 289},
    # Extended April 2018 activity surfaced by the v4 sweep (per-day table): strong,
    # threshold-robust detections beyond the 13th. SUSPECTED — no ref (retain% would be
    # against an unverified count); pending Gradio review by J. Ryan. See CLAUDE.md #14.
    {"name": "Apr 13–25 2018 — extended window (SUSPECTED)",  "start": "2018-04-13", "end": "2018-04-25", "truth": "present"},
    {"name": "Apr 18 2018 — strong day (SUSPECTED)",          "start": "2018-04-18", "end": "2018-04-18", "truth": "present"},
    {"name": "Apr 23–25 2018 — cluster (SUSPECTED)",          "start": "2018-04-23", "end": "2018-04-25", "truth": "present"},
    {"name": "May 12 2018 — event (CONFIRMED)",               "start": "2018-05-12", "end": "2018-05-12", "truth": "present", "ref": 190},
    {"name": "May 14 2018 — secondary (probable)",            "start": "2018-05-14", "end": "2018-05-14", "truth": "present", "ref": 45},
    {"name": "Oct 05–12 2020 — hunt cluster (SILENT)",        "start": "2020-10-05", "end": "2020-10-12", "truth": "absent",  "ref": 0},
    {"name": "October 2020 — full month (SILENT)",            "start": "2020-10-01", "end": "2020-10-31", "truth": "absent",  "ref": 0},
    {"name": "Apr 17–24 2026 — CA51A/CA50B window (no vocal)", "start": "2026-04-17", "end": "2026-04-24", "truth": "absent",  "ref": 0},
    {"name": "April 2026 — full month (humpback FP)",         "start": "2026-04-01", "end": "2026-04-30", "truth": "absent",  "ref": 0},
]


def parse_thresholds(s):
    return [float(x) for x in s.split(",") if x.strip() != ""]


def extract_date(filename, rx):
    m = rx.search(filename)
    if not m:
        return None
    y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
    try:
        return date(y, mo, d)
    except ValueError:
        return None


def load_orca_rows(csv_paths, orca_label, rx):
    """Return list of (date, score) for orca_call rows across all CSVs, plus diagnostics."""
    rows = []
    n_total = 0
    n_orca = 0
    n_nodate = 0
    label_col = score_col = fname_col = None
    for path in csv_paths:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            fields = reader.fieldnames or []
            # resolve columns once (schema-tolerant)
            label_col = next((c for c in ("label", "Label") if c in fields), None)
            score_col = next((c for c in ("logits", "logit", "score") if c in fields), None)
            fname_col = next((c for c in ("filename", "recording_id", "file") if c in fields), None)
            if not (label_col and score_col and fname_col):
                sys.exit(f"[{os.path.basename(path)}] could not find label/score/filename "
                         f"columns in header: {fields}")
            for r in reader:
                n_total += 1
                if r[label_col] != orca_label:
                    continue
                n_orca += 1
                d = extract_date(r[fname_col], rx)
                if d is None:
                    n_nodate += 1
                    continue
                try:
                    rows.append((d, float(r[score_col])))
                except (TypeError, ValueError):
                    n_nodate += 1
    return rows, {"n_total": n_total, "n_orca": n_orca, "n_nodate": n_nodate}


def count_in_region(rows, start, end, thr):
    return sum(1 for (d, s) in rows if start <= d <= end and s >= thr)


def score(rows, thresholds, regions):
    results = []
    for reg in regions:
        s = datetime.strptime(reg["start"], "%Y-%m-%d").date()
        e = datetime.strptime(reg["end"], "%Y-%m-%d").date()
        counts = {t: count_in_region(rows, s, e, t) for t in thresholds}
        results.append({**reg, "counts": counts})
    return results


def print_table(results, thresholds):
    wname = max(len(r["name"]) for r in results) + 2
    header = f"{'region':<{wname}}{'truth':>9}" + "".join(f"{('T='+format(t,'.2f')):>10}" for t in thresholds)
    print("\n" + "=" * len(header))
    print("ORCA DETECTIONS BY REGION AND THRESHOLD  (v4)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in results:
        line = f"{r['name']:<{wname}}{r['truth']:>9}"
        for t in thresholds:
            c = r["counts"][t]
            line += f"{c:>10}"
        print(line)
        # interpretation line
        if r["truth"] == "present" and r.get("ref"):
            ret = {t: (100.0 * r["counts"][t] / r["ref"]) for t in thresholds}
            interp = f"  retain% vs {r['ref']}:" + "".join(f"{ret[t]:>9.0f}%" for t in thresholds)
            print(f"{'':<{wname}}{'':>9}{interp}")
        elif r["truth"] == "absent":
            print(f"{'':<{wname}}{'':>9}  (all counts above are FALSE POSITIVES — want 0)")
    print("-" * len(header))
    print("PRESENT rows: higher retain% = fewer false negatives.  "
          "ABSENT rows: lower count = fewer false positives.")
    print("Pick the highest threshold that keeps PRESENT retain% high while ABSENT → 0.")


def print_by_day(rows, thresholds, regions):
    """Per-day orca counts within each PRESENT region's month, to show event structure."""
    present_months = set()
    for reg in regions:
        if reg["truth"] == "present":
            s = datetime.strptime(reg["start"], "%Y-%m-%d").date()
            present_months.add((s.year, s.month))
    if not present_months:
        return
    by_day = defaultdict(lambda: defaultdict(int))
    for (d, sc) in rows:
        if (d.year, d.month) in present_months:
            for t in thresholds:
                if sc >= t:
                    by_day[d][t] += 1
    print("\n" + "-" * 60)
    print("PER-DAY orca counts (present-event months only)")
    print("-" * 60)
    hdr = f"{'date':<12}" + "".join(f"{('T='+format(t,'.2f')):>10}" for t in thresholds)
    print(hdr)
    for d in sorted(by_day):
        line = f"{str(d):<12}" + "".join(f"{by_day[d][t]:>10}" for t in thresholds)
        print(line)


def write_summary(path, results, thresholds):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["region", "truth", "ref"] + [f"det_T{t:.2f}" for t in thresholds]
                   + [f"retain_pct_T{t:.2f}" for t in thresholds])
        for r in results:
            dets = [r["counts"][t] for t in thresholds]
            rets = [("" if r["truth"] != "present" or not r.get("ref")
                     else round(100.0 * r["counts"][t] / r["ref"], 1)) for t in thresholds]
            w.writerow([r["name"], r["truth"], r.get("ref", "")] + dets + rets)
    print(f"\nWrote {path}")


# ── self-test on synthetic CSVs mimicking the real schema ────────────────────────────
def selftest():
    import tempfile
    rng = __import__("random").Random(0)
    tmp = tempfile.mkdtemp()

    def make(fname, rows):
        p = os.path.join(tmp, fname)
        with open(p, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["idx", "project", "filename", "window_start", "window_end", "label", "logits"])
            for i, (mars, lab, sc) in enumerate(rows):
                w.writerow([i, "", mars, 0.0, 5.0, lab, round(sc, 5)])
        return p

    def mars(dt):  # MARS_YYYYMMDD_HHMMSS.wav style
        return f"MARS_{dt}_120000.wav"

    apr = ([ (mars("20180413"), "orca_call", rng.uniform(1.5, 4.0)) for _ in range(260) ]
           + [ (mars("20180413"), "orca_call", rng.uniform(0.0, 1.2)) for _ in range(29) ]  # weak
           + [ (mars("20180410"), "humpback_song", 2.0) for _ in range(50) ])
    may = ([ (mars("20180512"), "orca_call", rng.uniform(1.2, 3.5)) for _ in range(170) ]
           + [ (mars("20180514"), "orca_call", rng.uniform(0.8, 2.5)) for _ in range(40) ])
    octd = [ (mars("20201007"), "orca_call", rng.uniform(0.0, 1.4)) for _ in range(22) ]      # FPs, low
    a26 = ([ (mars("20260421"), "orca_call", rng.uniform(0.0, 1.3)) for _ in range(95) ]      # humpback FP, low
           + [ (mars("20260419"), "orca_call", rng.uniform(0.0, 0.9)) for _ in range(30) ])

    paths = [make("apr2018.csv", apr), make("may2018.csv", may),
             make("oct2020.csv", octd), make("apr2026.csv", a26)]

    rx = re.compile(DATE_REGEX_DEFAULT)
    rows, diag = load_orca_rows(paths, ORCA_LABEL_DEFAULT, rx)
    thresholds = [0.0, 1.0, 1.16, 1.5, 2.0]
    results = score(rows, thresholds, REGIONS)
    print(f"diagnostics: {diag}")
    print_table(results, thresholds)
    print_by_day(rows, thresholds, REGIONS)

    # invariants
    ok = True
    apr13 = next(r for r in results if r["name"].startswith("Apr 13"))
    oct_full = next(r for r in results if r["name"].startswith("October 2020"))
    a26_full = next(r for r in results if r["name"].startswith("April 2026"))
    if not (apr13["counts"][0.0] >= apr13["counts"][2.0]):
        print("FAIL: present counts should be monotonic non-increasing in T"); ok = False
    if not (apr13["counts"][1.5] >= 200):  # confirmed event retains strongly
        print("FAIL: Apr 13 event should retain most detections at T=1.5"); ok = False
    if not (oct_full["counts"][2.0] == 0 and a26_full["counts"][2.0] == 0):
        print("FAIL: absent-region FPs should hit 0 at T=2.0"); ok = False
    if not (a26_full["counts"][0.0] > a26_full["counts"][1.5]):
        print("FAIL: April 2026 FPs should collapse as T rises"); ok = False
    print("\nSELFTEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", help="dir containing the per-month inference CSVs")
    ap.add_argument("--pattern", default="*_v4_orcaval.csv", help="glob for CSVs in --results-dir")
    ap.add_argument("--csv", nargs="+", help="explicit CSV paths (instead of --results-dir/--pattern)")
    ap.add_argument("--thresholds", default="0.0,1.0,1.16,1.5,2.0")
    ap.add_argument("--orca-label", default=ORCA_LABEL_DEFAULT)
    ap.add_argument("--date-regex", default=DATE_REGEX_DEFAULT,
                    help="regex with 3 groups (Y,M,D) to pull the date from filename")
    ap.add_argument("--by-day", action="store_true", help="also print per-day counts for event months")
    ap.add_argument("--out-summary", help="write a summary CSV")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(selftest())

    if args.csv:
        paths = args.csv
    elif args.results_dir:
        paths = sorted(glob.glob(os.path.join(args.results_dir, args.pattern)))
    else:
        ap.error("provide --csv or --results-dir (or --selftest)")
    if not paths:
        sys.exit("No CSVs found.")
    print("CSVs:", *[os.path.basename(p) for p in paths])

    rx = re.compile(args.date_regex)
    thresholds = parse_thresholds(args.thresholds)
    rows, diag = load_orca_rows(paths, args.orca_label, rx)
    print(f"orca rows: {diag['n_orca']} of {diag['n_total']} total"
          + (f"  ({diag['n_nodate']} dropped: no parseable date)" if diag["n_nodate"] else ""))
    if diag["n_orca"] and not rows:
        sys.exit("Found orca rows but none had a parseable date — check --date-regex "
                 "against your filenames.")
    results = score(rows, thresholds, REGIONS)
    print_table(results, thresholds)
    if args.by_day:
        print_by_day(rows, thresholds, REGIONS)
    if args.out_summary:
        write_summary(args.out_summary, results, thresholds)


if __name__ == "__main__":
    main()
