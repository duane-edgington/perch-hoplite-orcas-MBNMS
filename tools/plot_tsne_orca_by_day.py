#!/usr/bin/env python3
"""
plot_tsne_orca_by_day.py — t-SNE of CONFIRMED orca_call embeddings, colored by day.

Purpose: explore Duane's ear-observation that orca calls sound different across the
confirmed April 2018 days (Apr 13 / 18 / 25) and May 12 2018. Produces two plots:
  1. April-only  (Apr 13, 18, 25)          — one month, cleaner acoustic environment
  2. All confirmed days (adds May 12 2018)   — May shown with a different marker shape

READ THIS BEFORE INTERPRETING (t-SNE caveats):
  - t-SNE is EXPLORATORY, not a test. Distances between clusters and cluster sizes are
    NOT meaningful; structure can appear/vanish with perplexity. Try a few --perplexity.
  - Perch V2 embeds SPECIES and deliberately collapses within-orca variation. If the days
    OVERLAP, that does NOT disprove the ear — it more likely means the embedding doesn't
    resolve pod/individual/call-type. A null result here is inconclusive, not negative.
  - CONFOUND: days differ in background/SNR/ship noise (whale-watch + CKWP boats co-occur
    with orca on event days). If days DO separate, rule out that it's environmental before
    calling it biological. Marker shape encodes month; consider a follow-up that encodes
    ship-noise co-occurrence per window.
  Real confirmation of call-type differences needs direct call analysis (Duane's domain),
  not this plot.

--------------------------------------------------------------------------------------
ADAPTER (>>> VERIFY — reuse tools/plot_tsne.py's loader)
    load_confirmed_orca() must return, for orca_call-labeled windows across the given DBs:
        emb   : float32 [N, D]   Perch V2 embeddings
        days  : [N] datetime.date  parsed from each window's source filename
        wids  : [N] int          window ids (for provenance)
    plot_tsne.py already loads labeled embeddings from these DBs — copy that exact loader
    and add the window->filename lookup (hoplite sqlite: recordings.filename joined to
    windows.recording_id; filename embeds YYYYMMDD). --selftest bypasses this entirely.
--------------------------------------------------------------------------------------

Examples
    python3 tools/plot_tsne_orca_by_day.py --selftest      # validate viz core, no DB
    python3 tools/plot_tsne_orca_by_day.py \
        --april-db /mnt/PAM_Analysis/perch-hoplite/db/MARS_20180401_20180430_32kHz_norm \
        --may-db   /mnt/PAM_Analysis/perch-hoplite/db/MARS_20180501_20180531_32kHz_norm \
        --out-dir  /mnt/PAM_Analysis/perch-hoplite/results \
        --perplexity 20
"""
import argparse
import os
import re
import sys
from datetime import date

import numpy as np

APRIL_DAYS = [date(2018, 4, 13), date(2018, 4, 18), date(2018, 4, 21), date(2018, 4, 25)]
MAY_DAYS = [date(2018, 5, 12)]
# NOTE: these are DEFAULTS only. Do not edit these to explore a new/different day set --
# use --confirmed-april-days / --confirmed-may-days on the command line instead (see CLI
# section below). Editing these constants silently changes what every future run produces,
# including anyone else's, which caused a real figure-mismatch incident (Aug 22 2026): a
# day added here for one exploratory run ended up in a "panel 8" render that was never
# supposed to include it. CLI flags make each run's day selection explicit and reproducible
# from the command alone, with no hidden state.
DAY_COLORS = {
    date(2018, 4, 13): "#1b9e77",   # confirmed Bigg's event
    date(2018, 4, 18): "#d95f02",   # confirmed bout
    date(2018, 4, 21): "#66a61e",   # confirmed (Aug 21 2026 resolution of the "pending" flag)
    date(2018, 4, 25): "#7570b3",   # confirmed cluster
    date(2018, 5, 12): "#e7298a",   # confirmed May event
}
_FALLBACK_COLOR_CYCLE = ["#999999", "#33a02c", "#e31a1c", "#ff7f00", "#6a3d9a", "#b15928"]


def _color_for_day(d, palette, fallback_cycle_state={}):
    """Look up a day's color in the given palette; if the day isn't listed (e.g. a new day
    passed via --confirmed-april-days that predates a DAY_COLORS entry), assign a stable
    fallback color instead of crashing or silently defaulting everything to gray. Caches by
    (palette id, day) so repeated lookups of the same unlisted day always return the same
    color within a run."""
    if d in palette:
        return palette[d]
    cache_key = (id(palette), d)
    if cache_key in fallback_cycle_state:
        return fallback_cycle_state[cache_key]
    idx = sum(1 for k in fallback_cycle_state if k[0] == id(palette))
    color = _FALLBACK_COLOR_CYCLE[idx % len(_FALLBACK_COLOR_CYCLE)]
    fallback_cycle_state[cache_key] = color
    print(f"  NOTE: {d.isoformat()} has no predefined color -- using fallback {color}. "
          f"Add it to DAY_COLORS/PRES_DAY_COLORS for a permanent, deliberate color choice.")
    return color
# Brighter day palette for the dark presentation theme (matches plot_tsne_orca_events.py
# slate background #1e293b). Distinct-by-day rather than the species green, since every
# point here is orca.
PRES_DAY_COLORS = {
    date(2018, 4, 13): "#38bdf8",   # sky
    date(2018, 4, 18): "#fbbf24",   # amber
    date(2018, 4, 21): "#4ade80",   # green (Aug 21 2026 addition)
    date(2018, 4, 25): "#c084fc",   # bright purple
    date(2018, 5, 12): "#fb7185",   # rose
}
_DATE_RE = re.compile(r"(20\d{2})(\d{2})(\d{2})")


# ======================================================================================
# ADAPTER — >>> VERIFY against tools/plot_tsne.py. --selftest bypasses this.
# ======================================================================================

def _day_from_filename(fname):
    m = _DATE_RE.search(fname or "")
    if not m:
        return None
    try:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
        return None


def load_confirmed_orca(db_dir, keep_days):
    """Return (emb[N,D], days[N], wids[N]) for orca_call windows whose filename-day is in
    keep_days. Loader mirrors tools/plot_tsne.py exactly (same annotations join + usearch
    index), then filters to label=='orca_call' (label_type != 2) on the requested days.
    """
    import sqlite3
    db_path = os.path.join(db_dir, "hoplite.sqlite")
    index_path = os.path.join(db_dir, "usearch.index")
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"hoplite.sqlite not found in {db_dir}")
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"usearch.index not found in {db_dir}")

    con = sqlite3.connect(db_path)
    rows = con.execute("""
        SELECT a.label, a.label_type, w.id, r.filename
        FROM annotations a
        JOIN recordings r ON r.id = a.recording_id
        JOIN windows w ON w.recording_id = a.recording_id
            AND w.offsets = a.offsets
        ORDER BY w.id
    """).fetchall()
    con.close()

    keep = set(keep_days)
    sel = []   # (window_id, day)
    for label, label_type, wid, fname in rows:
        if label_type == 2:        # weak negative — never an orca positive
            continue
        if label != "orca_call":
            continue
        d = _day_from_filename(fname)
        if d in keep:
            sel.append((wid, d))
    if not sel:
        return np.empty((0, 0), np.float32), np.array([], object), np.array([])

    from usearch.index import Index
    index = Index.restore(index_path, view=True)
    emb, days, wids = [], [], []
    for wid, d in sel:
        try:
            vec = index[wid]
        except Exception:
            continue
        emb.append(np.array(vec, dtype=np.float32)); days.append(d); wids.append(wid)
    return np.stack(emb, axis=0), np.array(days, object), np.array(wids)


# ======================================================================================
# VIZ CORE — validated by --selftest
# ======================================================================================

def run_tsne(emb, perplexity=None, seed=42):
    """t-SNE with the same geometry as tools/plot_tsne.py: unit-normalize + cosine metric,
    so this plot is comparable to the repo's other t-SNEs."""
    from sklearn.manifold import TSNE
    n = emb.shape[0]
    if perplexity is None:
        perplexity = max(5, min(30, (n - 1) // 3))
    perplexity = min(perplexity, n - 1)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb_norm = emb / np.maximum(norms, 1e-8)
    tsne = TSNE(n_components=2, perplexity=perplexity, init="pca",
                learning_rate="auto", random_state=seed, metric="cosine")
    return tsne.fit_transform(emb_norm), perplexity


def plot_by_day(coords, days, title, out_png, month_marker=False, style="analysis", dpi=150):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    uniq = sorted(set(days))
    if style == "presentation":
        # Dark theme matching tools/plot_tsne_orca_events.py
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.patch.set_facecolor("#0f172a")
        ax.set_facecolor("#1e293b")
        colors = PRES_DAY_COLORS
        for d in uniq:
            mask = np.array([x == d for x in days])
            marker = ("^" if d.month == 5 else "o") if month_marker else "o"
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       s=55, alpha=0.9, edgecolors="none",
                       c=_color_for_day(d, colors), marker=marker,
                       label=f"{d.isoformat()} (n={int(mask.sum())})")
        ax.set_title(title, color="#e2e8f0", fontsize=13, pad=12)
        ax.set_xlabel("t-SNE dim 1", color="#94a3b8", fontsize=9)
        ax.set_ylabel("t-SNE dim 2", color="#94a3b8", fontsize=9)
        ax.tick_params(colors="#94a3b8", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#334155")
        ax.legend(framealpha=0.15, facecolor="#0f172a", edgecolor="#334155",
                  labelcolor="#e2e8f0", fontsize=9, loc="best")
        ax.text(0.01, 0.01,
                f"n={len(days)} confirmed orca windows  |  Perch V2 1536-dim → t-SNE 2D  |  exploratory",
                transform=ax.transAxes, fontsize=7, color="#475569", va="bottom")
        fig.tight_layout()
        fig.savefig(out_png, dpi=dpi, bbox_inches="tight", facecolor="#0f172a")
        plt.close(fig)
        return out_png

    # Default: light analysis theme (readable while iterating, with full caveat)
    fig, ax = plt.subplots(figsize=(8, 7))
    for d in uniq:
        mask = np.array([x == d for x in days])
        marker = ("^" if d.month == 5 else "o") if month_marker else "o"
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   s=42, alpha=0.75, edgecolors="white", linewidths=0.4,
                   c=_color_for_day(d, DAY_COLORS), marker=marker,
                   label=f"{d.isoformat()} (n={int(mask.sum())})")
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    ax.text(0.5, -0.09,
            "Exploratory — t-SNE distances/sizes are not meaningful; overlap is inconclusive, "
            "not a null result.",
            transform=ax.transAxes, ha="center", va="top", fontsize=7, color="#666")
    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_png


def make_plots(april_emb, april_days, april_wids,
               may_emb, may_days, may_wids,
               out_dir, perplexity, seed, style="analysis", dpi=150):
    os.makedirs(out_dir, exist_ok=True)
    written = []
    sfx = "_pres" if style == "presentation" else ""
    px_tag = f"px{int(perplexity)}" if perplexity else "pxauto"
    dpi_tag = "" if dpi == 150 else f"_dpi{dpi}"

    # Plot 1 — April only
    n_april_days = len(set(april_days.tolist()))
    coords_a, px_a = run_tsne(april_emb, perplexity, seed)
    p1 = plot_by_day(coords_a, april_days,
                     f"Confirmed orca calls — April 2018 by day (t-SNE, perplexity={px_a}, n={len(april_days)})",
                     os.path.join(out_dir, f"tsne_orca_by_day_april2018_{n_april_days}days_{px_tag}{sfx}{dpi_tag}.png"),
                     style=style, dpi=dpi)
    written.append(p1)

    # Plot 2 — all confirmed days (April + May 12); May as triangle
    all_emb = np.concatenate([april_emb, may_emb], axis=0)
    all_days = np.concatenate([april_days, may_days])
    n_days = len(set(all_days.tolist()))
    coords_all, px_all = run_tsne(all_emb, perplexity, seed)
    p2 = plot_by_day(coords_all, all_days,
                     f"Confirmed orca calls — {n_days} days Apr+May 2018 (t-SNE, perplexity={px_all}, n={len(all_days)})",
                     os.path.join(out_dir, f"tsne_orca_by_day_{n_days}days_{px_tag}{sfx}{dpi_tag}.png"),
                     month_marker=True, style=style, dpi=dpi)
    written.append(p2)
    return written


# ======================================================================================
# SELFTEST — synthetic 1536-dim embeddings, exercises t-SNE + both plots
# ======================================================================================

def selftest():
    rng = np.random.default_rng(0)
    D = 1536

    def cluster(center_seed, n, spread=1.0):
        base = rng.normal(0, 1, D)
        base = base / np.linalg.norm(base) * center_seed
        return base + rng.normal(0, spread, (n, D)).astype(np.float32)

    # four April "days" with modest separation, plus a more distinct May day
    a13 = cluster(6.0, 60); a18 = cluster(6.2, 45); a21 = cluster(6.1, 55); a25 = cluster(6.4, 50)
    m12 = cluster(9.0, 40)
    april_emb = np.concatenate([a13, a18, a21, a25]).astype(np.float32)
    april_days = np.array([date(2018, 4, 13)] * 60 + [date(2018, 4, 18)] * 45
                          + [date(2018, 4, 21)] * 55 + [date(2018, 4, 25)] * 50, dtype=object)
    april_wids = np.arange(len(april_days))
    may_emb = m12.astype(np.float32)
    may_days = np.array([date(2018, 5, 12)] * 40, dtype=object)
    may_wids = np.arange(len(may_days)) + 1000

    import tempfile
    out = tempfile.mkdtemp()
    written = []
    for st in ('analysis', 'presentation'):
        written += make_plots(april_emb, april_days, april_wids,
                              may_emb, may_days, may_wids, out, perplexity=None, seed=1, style=st)

    ok = True
    for p in written:
        if not (os.path.exists(p) and os.path.getsize(p) > 5000):
            print(f"FAIL: plot not written or too small: {p}"); ok = False
    # day-grouping sanity: filename parser round-trips
    if _day_from_filename("MARS_20180418_113912_resampled_32kHz.wav") != date(2018, 4, 18):
        print("FAIL: filename->day parse"); ok = False
    if _day_from_filename("no_date_here.wav") is not None:
        print("FAIL: bad filename should give None"); ok = False
    print("Wrote:", *[os.path.basename(p) for p in written])
    print("SELFTEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


# ======================================================================================
# CLI
# ======================================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--april-db", help="April 2018 norm DB dir")
    ap.add_argument("--may-db", help="May 2018 norm DB dir")
    ap.add_argument("--out-dir", default="/mnt/PAM_Analysis/perch-hoplite/results")
    ap.add_argument("--perplexity", type=float, default=None,
                    help="t-SNE perplexity (default auto ~min(30,(n-1)/3)); try a few")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--style", choices=["analysis", "presentation"], default="analysis",
                    help="analysis=light readable (default); presentation=dark theme for slides")
    ap.add_argument("--dpi", type=int, default=150,
                    help="output image resolution (default 150; use 300 for print/poster quality)")
    ap.add_argument("--confirmed-april-days", default=None,
                    help="comma-separated YYYY-MM-DD dates overriding the default April day "
                         "list for THIS RUN ONLY (does not touch the APRIL_DAYS constant). "
                         f"Default: {','.join(d.isoformat() for d in APRIL_DAYS)}. "
                         "Use this instead of editing APRIL_DAYS in the source -- e.g. to "
                         "reproduce a figure from before a day was added, pass the exact "
                         "historical day list explicitly.")
    ap.add_argument("--confirmed-may-days", default=None,
                    help="comma-separated YYYY-MM-DD dates overriding the default May day "
                         f"list for THIS RUN ONLY. Default: {','.join(d.isoformat() for d in MAY_DAYS)}.")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(selftest())
    if not args.april_db:
        ap.error("--april-db required (or --selftest)")

    def _parse_days(s):
        out = []
        for tok in s.split(","):
            tok = tok.strip()
            if not tok:
                continue
            y, m, d = (int(x) for x in tok.split("-"))
            out.append(date(y, m, d))
        return out

    active_april_days = _parse_days(args.confirmed_april_days) if args.confirmed_april_days else APRIL_DAYS
    active_may_days = _parse_days(args.confirmed_may_days) if args.confirmed_may_days else MAY_DAYS
    if args.confirmed_april_days:
        print(f"  Using explicit April day list (overrides default): "
              f"{[d.isoformat() for d in active_april_days]}")
    if args.confirmed_may_days:
        print(f"  Using explicit May day list (overrides default): "
              f"{[d.isoformat() for d in active_may_days]}")

    print("Loading confirmed orca embeddings...")
    april_emb, april_days, april_wids = load_confirmed_orca(args.april_db, active_april_days)
    print(f"  April: {len(april_days)} orca windows across "
          f"{sorted(set(d.isoformat() for d in april_days))}")
    if args.may_db:
        may_emb, may_days, may_wids = load_confirmed_orca(args.may_db, active_may_days)
        print(f"  May:   {len(may_days)} orca windows")
    else:
        may_emb = np.empty((0, april_emb.shape[1]), np.float32)
        may_days = np.array([], dtype=object); may_wids = np.array([])

    written = make_plots(april_emb, april_days, april_wids,
                         may_emb, may_days, may_wids,
                         args.out_dir, args.perplexity, args.seed, style=args.style, dpi=args.dpi)
    print("Wrote:")
    for p in written:
        print(" ", p)
    print("\nRegister each with tools/register_figure.py (--type tsne_plot) per the "
          "Figure Provenance workflow before committing.")


if __name__ == "__main__":
    main()
