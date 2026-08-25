#!/usr/bin/env python3
"""plot_monthly.py
Plot a full-month detection timeline from a multi-day inference CSV.
Shows detections per day for each class, plus a UTC heatmap.

Usage:
    python3 plot_monthly.py \
        --input /mnt/PAM_Analysis/duane_scratch/perch_hoplite/results/MARS_20180401_20180430_v5_clean_detections.csv \
        --output-dir /mnt/PAM_Analysis/duane_scratch/perch_hoplite/results \
        --title "April 2018 — MARS Hydrophone — v5_clean"
"""

import argparse
import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from pathlib import Path


LABEL_COLORS = {
    "orca_call":     "#16a34a",
    "humpback_song": "#d97706",
    "fin_whale_call":"#2563eb",
    "dolphin_call":  "#9333ea",
    "ship_noise":    "#0891b2",
    "other":         "#ea580c",
    "negative":      "#6b7280",
}
DEFAULT_COLOR = "#94a3b8"


def parse_datetime(filename, offset_s):
    m = re.search(r'MARS_(\d{8})_(\d{6})', str(filename))
    if not m:
        return None
    try:
        dt = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
        return dt + timedelta(seconds=float(offset_s))
    except Exception:
        return None


def load_csv(input_csv):
    df = pd.read_csv(input_csv)
    # Deduplicate on (idx, label) — each unique window+label pair counts once.
    # Deduplicating on idx alone would collapse multi-label detections
    # (e.g. a window detected as both orca_call and humpback_song) into one row.
    before = len(df)
    df = df.drop_duplicates(subset=["idx", "label"])
    removed = before - len(df)
    if removed:
        print(f"  Note: removed {removed} duplicate rows after deduplication on (idx, label)")
    print(f"Loaded {len(df)} detections from {Path(input_csv).name}")
    df["datetime_utc"] = [
        parse_datetime(f, s)
        for f, s in zip(df["filename"], df["window_start"])
    ]
    df = df.dropna(subset=["datetime_utc"])
    df["date"]     = df["datetime_utc"].dt.date
    df["utc_hour"] = df["datetime_utc"].dt.hour + df["datetime_utc"].dt.minute / 60.0
    # Extract day from filename directly — avoids UTC midnight bleed into next month
    def _day_from_fname(fname):
        import re as _r
        m = _r.search(r'MARS_(\d{6})(\d{2})_', str(fname))
        return int(m.group(2)) if m else 0
    df["day_of_month"] = df["filename"].apply(_day_from_fname)
    return df


def plot_monthly(df, output_dir, title):
    labels = sorted(df["label"].unique())
    dates  = sorted(df["date"].unique())
    n_days = len(dates)

    # ── Figure 1: Daily detection counts per class ────────────────────────
    fig, axes = plt.subplots(len(labels), 1,
                             figsize=(14, 3 * len(labels)),
                             gridspec_kw={"hspace": 0.5})
    if len(labels) == 1:
        axes = [axes]
    fig.patch.set_facecolor("#0f172a")
    fig.suptitle(f"{title}\nDetections per day", color="#e2e8f0",
                 fontsize=12, y=0.99)

    day_nums = [d.day for d in dates]

    for ax, lbl in zip(axes, labels):
        ax.set_facecolor("#1e293b")
        sub = df[df["label"] == lbl]
        counts = sub.groupby("day_of_month").size()
        color  = LABEL_COLORS.get(lbl, DEFAULT_COLOR)

        all_days = list(range(1, 32))
        all_counts = [counts.get(d, 0) for d in all_days]
        present = [d for d in all_days if counts.get(d, 0) > 0]
        present_counts = [counts.get(d, 0) for d in present]

        ax.bar(present, present_counts, color=color, alpha=0.85, width=0.7)
        ax.set_xlim(0.5, 31.5)
        ax.set_ylabel("Detections", color="#94a3b8", fontsize=8)
        ax.set_title(f"{lbl}  (total: {len(sub):,})",
                     color="#e2e8f0", fontsize=10)
        ax.tick_params(colors="#94a3b8", labelsize=8)
        ax.set_xticks(range(1, 32))
        ax.set_xticklabels([str(d) for d in range(1, 32)], fontsize=7,
                           color="#94a3b8")
        for spine in ax.spines.values():
            spine.set_edgecolor("#334155")

    # Derive month label from title if possible
    import re as _re2
    _month_match = _re2.search(r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})', title)
    _month_label = f"{_month_match.group(1)} {_month_match.group(2)} UTC" if _month_match else "UTC"
    axes[-1].set_xlabel(f"Day of month ({_month_label})",
                        color="#94a3b8", fontsize=9)

    # Derive prefix from input CSV filename
    import re as _re
    m = _re.search(r'MARS_(\d{8})_(\d{8})', title + " " + output_dir)
    csv_stem = Path(output_dir).name  # fallback
    # Build prefix from title
    prefix = title.replace(" ", "_").replace("—", "").replace("__", "_").strip("_")
    prefix = _re.sub(r'[^A-Za-z0-9_]', '', prefix)[:40]
    out1 = os.path.join(output_dir, f"{prefix}_monthly_counts.png")
    fig.savefig(out1, dpi=150, bbox_inches="tight", facecolor="#0f172a")
    plt.close(fig)
    print(f"Saved: {out1}")

    # ── Figure 2: UTC heatmap — detections by day × hour ─────────────────
    fig2, axes2 = plt.subplots(len(labels), 1,
                               figsize=(16, 2.8 * len(labels)),
                               gridspec_kw={"hspace": 0.6})
    if len(labels) == 1:
        axes2 = [axes2]
    fig2.patch.set_facecolor("#0f172a")
    fig2.suptitle(f"{title}\nDetection heatmap (day × UTC hour)",
                  color="#e2e8f0", fontsize=12, y=0.99)

    for ax, lbl in zip(axes2, labels):
        ax.set_facecolor("#1e293b")
        sub = df[df["label"] == lbl].copy()
        sub["hour_bin"] = sub["utc_hour"].apply(lambda h: int(h))

        # Build 31×24 grid
        grid = np.zeros((31, 24))
        for _, row in sub.iterrows():
            d = int(row["day_of_month"]) - 1
            h = int(row["hour_bin"])
            if 0 <= d < 31 and 0 <= h < 24:
                grid[d, h] += 1

        color = LABEL_COLORS.get(lbl, DEFAULT_COLOR)
        # Build custom colormap from black to label color
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list(
            lbl, ["#111827", color], N=256)

        im = ax.imshow(grid, aspect="auto", cmap=cmap,
                       origin="upper", interpolation="nearest")
        plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02,
                     label="Detections").ax.tick_params(
            colors="#94a3b8", labelsize=7)

        ax.set_yticks(range(0, 31, 2))
        ax.set_yticklabels([str(d+1) for d in range(0, 31, 2)],
                           fontsize=7, color="#94a3b8")
        ax.set_xticks(range(0, 24, 2))
        ax.set_xticklabels(
            [f"{h:02d}\n({(h-7)%24:02d}P)" for h in range(0, 24, 2)],
            fontsize=6.5, color="#94a3b8")
        ax.set_ylabel("Day (April UTC)", color="#94a3b8", fontsize=8)
        ax.set_title(f"{lbl}  (total: {len(sub):,})",
                     color="#e2e8f0", fontsize=10)
        for spine in ax.spines.values():
            spine.set_edgecolor("#334155")

    axes2[-1].set_xlabel("UTC Hour (PDT = UTC−7)",
                         color="#94a3b8", fontsize=9)

    out2 = os.path.join(output_dir, f"{prefix}_heatmap.png")
    fig2.savefig(out2, dpi=150, bbox_inches="tight", facecolor="#0f172a")
    plt.close(fig2)
    print(f"Saved: {out2}")

    # Summary
    print(f"\nSummary — {title}")
    print(f"  Days covered: {n_days}")
    for lbl in labels:
        n = len(df[df["label"] == lbl])
        days_active = df[df["label"] == lbl]["day_of_month"].nunique()
        print(f"  {lbl:20s}: {n:6,} detections across {days_active} days")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input",      required=True,
                    help="Inference detections CSV (full month)")
    ap.add_argument("--output-dir", required=True,
                    help="Directory for output PNGs")
    ap.add_argument("--title",
                    default="MARS Hydrophone — Monthly Detection Timeline")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    df = load_csv(args.input)
    plot_monthly(df, args.output_dir, args.title)


if __name__ == "__main__":
    main()
