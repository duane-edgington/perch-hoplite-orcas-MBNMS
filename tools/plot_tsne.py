#!/usr/bin/env python3
"""plot_tsne.py
Generate a t-SNE plot of Perch V2 embeddings from a hoplite DB,
colored by annotation label.

Shows only labeled windows (annotations in the DB). Unlabeled windows
are excluded — they would just add noise to the visualization.

Usage:
    python3 plot_tsne.py \
        --db-dir /mnt/PAM_Analysis/duane_scratch/perch_hoplite/db/MARS_20180413_20180413_32kHz \
        --output /mnt/PAM_Analysis/duane_scratch/perch_hoplite/results/tsne_MARS_20180413.png \
        [--perplexity 30] [--n-iter 1000] [--title "April 13 2018"]

    # Multiple DBs combined (e.g. April 13 + April 30 + May 2):
    python3 plot_tsne.py \
        --db-dir /path/to/db1 /path/to/db2 /path/to/db3 \
        --output /path/to/tsne_combined.png \
        --title "April–May 2018 combined"
"""

import argparse
import os
import struct
import sqlite3
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


# Label colors matching the Gradio annotation interface
LABEL_COLORS = {
    "orca_call":     "#16a34a",   # green
    "humpback_song": "#d97706",   # amber
    "fin_whale_call":"#2563eb",   # blue
    "dolphin_call":  "#9333ea",   # purple
    "ship_noise":    "#0891b2",   # teal
    "other":         "#f43f5e",   # rose
    "negative":      "#6b7280",   # gray
    "background":    "#6b7280",   # gray
}
DEFAULT_COLOR = "#94a3b8"


def load_labeled_embeddings(db_dir: str):
    """Load embeddings and labels for all annotated windows in a DB."""
    db_path = os.path.join(db_dir, "hoplite.sqlite")
    index_path = os.path.join(db_dir, "usearch.index")

    if not os.path.exists(db_path):
        raise FileNotFoundError(f"hoplite.sqlite not found in {db_dir}")
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"usearch.index not found in {db_dir}")

    # Load annotations
    con = sqlite3.connect(db_path)
    rows = con.execute("""
        SELECT a.label, a.label_type, w.id, r.filename,
               w.offsets
        FROM annotations a
        JOIN recordings r ON r.id = a.recording_id
        JOIN windows w ON w.recording_id = a.recording_id
            AND w.offsets = a.offsets
        ORDER BY a.label, w.id
    """).fetchall()
    con.close()

    if not rows:
        print(f"  WARNING: no annotations found in {db_dir}")
        return np.array([]), [], []

    window_ids = [r[2] for r in rows]
    labels     = []
    for label, label_type, wid, fname, off_blob in rows:
        if label_type == 2:
            labels.append("negative")
        else:
            labels.append(label)

    # Load embeddings from USearch index
    try:
        from usearch.index import Index
        index = Index.restore(index_path, view=True)
        embs = []
        for wid in window_ids:
            try:
                vec = index[wid]
                embs.append(np.array(vec, dtype=np.float32))
            except Exception:
                embs.append(None)
        # Filter out any None (missing embeddings)
        valid = [(e, l) for e, l in zip(embs, labels) if e is not None]
        if not valid:
            return np.array([]), [], []
        embs, labels = zip(*valid)
        embeddings = np.stack(embs, axis=0)
    except ImportError:
        raise ImportError("usearch not installed. Run: pip install usearch")

    print(f"  Loaded {len(labels)} labeled embeddings from {Path(db_dir).name}")
    from collections import Counter
    for lbl, cnt in sorted(Counter(labels).items()):
        print(f"    {lbl}: {cnt}")

    return embeddings, list(labels), window_ids


def run_tsne(embeddings: np.ndarray, perplexity: int, n_iter: int,
             random_state: int = 42):
    """Run t-SNE dimensionality reduction."""
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        raise ImportError("scikit-learn not installed. Run: pip install scikit-learn")

    print(f"\nRunning t-SNE on {embeddings.shape[0]} embeddings "
          f"({embeddings.shape[1]} dims) → 2D ...")
    print(f"  perplexity={perplexity}, n_iter={n_iter}")

    # Normalize embeddings to unit length before t-SNE
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings_norm = embeddings / np.maximum(norms, 1e-8)

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        max_iter=n_iter,
        random_state=random_state,
        verbose=1,
        metric="cosine",
    )
    coords = tsne.fit_transform(embeddings_norm)
    print(f"  t-SNE complete. KL divergence: {tsne.kl_divergence_:.4f}")
    return coords


def plot_tsne(coords: np.ndarray, labels: list, output_path: str,
              title: str = "Perch V2 Embeddings — t-SNE", dpi: int = 150):
    """Plot t-SNE coordinates colored by label."""
    unique_labels = sorted(set(labels))
    n_labels = len(unique_labels)

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")

    for lbl in unique_labels:
        mask = [i for i, l in enumerate(labels) if l == lbl]
        x = coords[mask, 0]
        y = coords[mask, 1]
        color = LABEL_COLORS.get(lbl, DEFAULT_COLOR)
        ax.scatter(x, y, c=color, label=f"{lbl} (n={len(mask)})",
                   alpha=0.75, s=35, linewidths=0.3,
                   edgecolors="white")

    ax.set_title(title, color="#e2e8f0", fontsize=13, pad=12)
    ax.set_xlabel("t-SNE dim 1", color="#94a3b8", fontsize=9)
    ax.set_ylabel("t-SNE dim 2", color="#94a3b8", fontsize=9)
    ax.tick_params(colors="#94a3b8", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#334155")

    legend = ax.legend(
        fontsize=9, facecolor="#1e293b", labelcolor="#e2e8f0",
        edgecolor="#334155", markerscale=1.4,
        loc="best",
    )

    # Annotation count
    ax.text(0.01, 0.01,
            f"Total: {len(labels)} labeled windows  |  "
            f"{n_labels} classes  |  "
            f"Perch V2 1536-dim → t-SNE 2D",
            transform=ax.transAxes,
            color="#64748b", fontsize=7.5, va="bottom")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight",
                facecolor="#0f172a")
    plt.close(fig)
    print(f"\nSaved: {output_path}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--db-dir", nargs="+", required=True,
        help="One or more hoplite DB directories to include")
    ap.add_argument("--output", required=True,
        help="Output PNG path")
    ap.add_argument("--title", default="Perch V2 Embeddings — t-SNE",
        help="Plot title")
    ap.add_argument("--perplexity", type=int, default=30,
        help="t-SNE perplexity (default 30; try 10-50)")
    ap.add_argument("--n-iter", type=int, default=1000,
        help="t-SNE iterations (default 1000)")
    ap.add_argument("--seed", type=int, default=42,
        help="Random seed for reproducibility")
    ap.add_argument("--dpi", type=int, default=150,
        help="output image resolution (default 150; use 300 for print/poster quality)")
    args = ap.parse_args()

    all_embeddings = []
    all_labels     = []

    for db_dir in args.db_dir:
        print(f"\nLoading: {db_dir}")
        embs, labels, _ = load_labeled_embeddings(db_dir)
        if len(embs) > 0:
            all_embeddings.append(embs)
            all_labels.extend(labels)

    if not all_embeddings:
        print("ERROR: no labeled embeddings found in any DB.")
        return 1

    embeddings = np.concatenate(all_embeddings, axis=0)
    print(f"\nTotal: {len(all_labels)} labeled embeddings from "
          f"{len(args.db_dir)} DB(s)")

    coords = run_tsne(embeddings, args.perplexity, args.n_iter, args.seed)
    plot_tsne(coords, all_labels, args.output, args.title, dpi=args.dpi)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
