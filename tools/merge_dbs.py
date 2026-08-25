#!/usr/bin/env python3
"""merge_dbs.py
Merge two Hoplite databases (SQLite + USearch) into a single combined DB.

Both DBs must have been built with the same model (same embedding_dim,
dtype, and metric). The combined DB contains all embeddings and all
annotations from both source DBs.

Usage:
    python3 merge_dbs.py \
        --db-a /path/to/db_a \
        --db-b /path/to/db_b \
        --output /path/to/combined_db \
        [--dry-run]

Example:
    python3 merge_dbs.py \
        --db-a /mnt/PAM_Analysis/duane_scratch/perch_hoplite/db/MARS_20180413_20180413_32kHz \
        --db-b /mnt/PAM_Analysis/duane_scratch/perch_hoplite/db/MARS_20180401_20180401_32kHz \
        --output /mnt/PAM_Analysis/duane_scratch/perch_hoplite/db/MARS_combined
"""

import argparse
import json
import os
import shutil
import sqlite3
import struct
import sys


def get_db_file(db_dir: str) -> str:
    p = os.path.join(db_dir, "hoplite.sqlite")
    if not os.path.isfile(p):
        raise FileNotFoundError(f"hoplite.sqlite not found in {db_dir}")
    return p


def get_index_file(db_dir: str) -> str:
    p = os.path.join(db_dir, "usearch.index")
    if not os.path.isfile(p):
        raise FileNotFoundError(f"usearch.index not found in {db_dir}")
    return p


def get_metadata(db_file: str, key: str):
    con = sqlite3.connect(db_file)
    row = con.execute(
        "SELECT value FROM hoplite_metadata WHERE key=?", (key,)
    ).fetchone()
    con.close()
    return json.loads(row[0]) if row else None


def create_output_db(output_dir: str, usearch_config: dict,
                     model_config: dict, audio_sources: dict):
    """Create the output directory and initialize a fresh SQLite DB."""
    os.makedirs(output_dir, exist_ok=True)
    logs_dir = os.path.join(output_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    db_file = os.path.join(output_dir, "hoplite.sqlite")
    if os.path.exists(db_file):
        os.remove(db_file)

    con = sqlite3.connect(db_file)
    con.executescript("""
        PRAGMA journal_mode = WAL;
        PRAGMA foreign_keys = ON;

        CREATE TABLE hoplite_metadata (
            key TEXT PRIMARY KEY, value TEXT NOT NULL);

        CREATE TABLE deployments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL, project TEXT NOT NULL,
            latitude REAL, longitude REAL,
            UNIQUE (name, project));

        CREATE TABLE recordings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL, datetime TEXT,
            deployment_id INTEGER REFERENCES deployments(id) ON DELETE CASCADE,
            UNIQUE (id, deployment_id),
            UNIQUE (filename, deployment_id));

        CREATE TABLE windows (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            recording_id INTEGER NOT NULL
                REFERENCES recordings(id) ON DELETE CASCADE,
            offsets FLOAT_LIST NOT NULL,
            UNIQUE (id, recording_id, offsets));

        CREATE TABLE annotations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            recording_id INTEGER NOT NULL
                REFERENCES recordings(id) ON DELETE CASCADE,
            offsets FLOAT_LIST NOT NULL,
            label TEXT NOT NULL, label_type INTEGER NOT NULL,
            provenance TEXT NOT NULL,
            UNIQUE (id, recording_id, offsets));

        CREATE INDEX idx_annotations ON annotations
            (recording_id, offsets, label, label_type, provenance);
        CREATE INDEX idx_labels ON annotations
            (label, label_type, provenance);
        CREATE INDEX idx_recordings_deployment_id ON recordings
            (deployment_id);
        CREATE INDEX idx_windows_recording_id ON windows
            (recording_id);
    """)

    for key, val in [
        ("usearch_config", usearch_config),
        ("model_config",   model_config),
        ("audio_sources",  audio_sources),
    ]:
        if val is not None:
            con.execute(
                "INSERT OR REPLACE INTO hoplite_metadata (key,value) VALUES (?,?)",
                (key, json.dumps(val))
            )
    con.commit()
    con.close()
    return db_file


def copy_db_contents(src_db: str, out_db: str,
                     window_id_offset: int) -> tuple[int, int]:
    """Copy all recordings, windows, and annotations from src to out.

    window_id_offset is added to all window IDs from src so they don't
    collide with windows already in out.

    Returns (windows_copied, annotations_copied).
    """
    src = sqlite3.connect(src_db)
    out = sqlite3.connect(out_db)
    out.execute("PRAGMA foreign_keys = OFF")

    dep_map = {}   # src deployment id -> out deployment id
    rec_map = {}   # src recording id  -> out recording id
    win_map = {}   # src window id     -> out window id

    # ── Deployments ──────────────────────────────────────────────────────────
    for dep_id, name, project, lat, lon in src.execute(
        "SELECT id, name, project, latitude, longitude FROM deployments"
    ):
        out.execute(
            "INSERT OR IGNORE INTO deployments (name, project, latitude, longitude)"
            " VALUES (?,?,?,?)", (name, project, lat, lon)
        )
        new_id = out.execute(
            "SELECT id FROM deployments WHERE name=? AND project=?",
            (name, project)
        ).fetchone()[0]
        dep_map[dep_id] = new_id

    # ── Recordings ───────────────────────────────────────────────────────────
    for rec_id, filename, dt, dep_id in src.execute(
        "SELECT id, filename, datetime, deployment_id FROM recordings"
    ):
        new_dep = dep_map.get(dep_id, dep_id)
        out.execute(
            "INSERT OR IGNORE INTO recordings (filename, datetime, deployment_id)"
            " VALUES (?,?,?)", (filename, dt, new_dep)
        )
        new_id = out.execute(
            "SELECT id FROM recordings WHERE filename=? AND deployment_id=?",
            (filename, new_dep)
        ).fetchone()[0]
        rec_map[rec_id] = new_id

    # ── Windows ──────────────────────────────────────────────────────────────
    win_count = 0
    for win_id, rec_id, offsets in src.execute(
        "SELECT id, recording_id, offsets FROM windows"
    ):
        new_rec = rec_map.get(rec_id, rec_id)
        new_win_id = win_id + window_id_offset
        out.execute(
            "INSERT OR IGNORE INTO windows (id, recording_id, offsets)"
            " VALUES (?,?,?)", (new_win_id, new_rec, offsets)
        )
        win_map[win_id] = new_win_id
        win_count += 1

    # ── Annotations ──────────────────────────────────────────────────────────
    ann_count = 0
    for rec_id, offsets, label, ltype, prov in src.execute(
        "SELECT recording_id, offsets, label, label_type, provenance"
        " FROM annotations"
    ):
        new_rec = rec_map.get(rec_id, rec_id)
        out.execute(
            "INSERT OR IGNORE INTO annotations"
            " (recording_id, offsets, label, label_type, provenance)"
            " VALUES (?,?,?,?,?)",
            (new_rec, offsets, label, ltype, prov)
        )
        ann_count += 1

    out.commit()
    out.execute("PRAGMA foreign_keys = ON")
    src.close()
    out.close()
    return win_count, ann_count


def merge_usearch_indexes(index_a: str, index_b: str,
                          output_index: str, window_id_offset: int,
                          usearch_config: dict):
    """Merge two USearch indexes into one.

    Reads all vectors from both indexes and writes them to a new index.
    Window IDs from index_b are offset by window_id_offset.
    """
    import numpy as np
    import usearch.index as ui

    cfg = usearch_config
    dim = cfg["embedding_dim"]
    dtype = cfg.get("dtype", "float16")
    metric = cfg.get("metric_name", "ip")
    np_dtype = np.float16 if dtype == "float16" else np.float32

    print(f"  Loading index A: {index_a}")
    idx_a = ui.Index(ndim=dim, metric=metric, dtype=dtype)
    idx_a.load(index_a)
    n_a = len(idx_a)
    print(f"    {n_a:,} vectors")

    print(f"  Loading index B: {index_b}")
    idx_b = ui.Index(ndim=dim, metric=metric, dtype=dtype)
    idx_b.load(index_b)
    n_b = len(idx_b)
    print(f"    {n_b:,} vectors")

    print(f"  Creating combined index ({n_a + n_b:,} vectors)...")
    idx_out = ui.Index(ndim=dim, metric=metric, dtype=dtype)

    # Copy index A vectors
    keys_a = np.asarray(list(idx_a.keys), dtype=np.int64)
    vecs_a = idx_a.get(keys_a.tolist())
    if isinstance(vecs_a, tuple):
        vecs_a = np.stack(vecs_a)
    idx_out.add(keys_a, vecs_a.astype(np_dtype))
    print(f"  Copied {n_a:,} vectors from A")

    # Copy index B vectors with offset IDs
    keys_b = np.asarray(list(idx_b.keys), dtype=np.int64)
    keys_b_offset = keys_b + window_id_offset
    vecs_b = idx_b.get(keys_b.tolist())
    if isinstance(vecs_b, tuple):
        vecs_b = np.stack(vecs_b)
    idx_out.add(keys_b_offset, vecs_b.astype(np_dtype))
    print(f"  Copied {n_b:,} vectors from B (IDs offset by {window_id_offset})")

    idx_out.save(output_index)
    print(f"  Saved combined index: {output_index}")
    return n_a + n_b


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--db-a", required=True, help="First DB directory (base)")
    ap.add_argument("--db-b", required=True, help="Second DB directory (to merge in)")
    ap.add_argument("--output", required=True, help="Output DB directory")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    db_a = get_db_file(args.db_a)
    db_b = get_db_file(args.db_b)
    idx_a = get_index_file(args.db_a)
    idx_b = get_index_file(args.db_b)

    # Validate configs match
    cfg_a = get_metadata(db_a, "usearch_config")
    cfg_b = get_metadata(db_b, "usearch_config")
    if cfg_a["embedding_dim"] != cfg_b["embedding_dim"]:
        print(f"ERROR: embedding_dim mismatch: {cfg_a['embedding_dim']} vs "
              f"{cfg_b['embedding_dim']}")
        sys.exit(1)
    print(f"embedding_dim: {cfg_a['embedding_dim']} ✓")

    # Count embeddings in A to compute offset for B
    con_a = sqlite3.connect(db_a)
    max_win_id = con_a.execute("SELECT MAX(id) FROM windows").fetchone()[0] or 0
    n_windows_a = con_a.execute("SELECT COUNT(*) FROM windows").fetchone()[0]
    n_ann_a = con_a.execute("SELECT COUNT(*) FROM annotations").fetchone()[0]
    con_a.close()

    con_b = sqlite3.connect(db_b)
    n_windows_b = con_b.execute("SELECT COUNT(*) FROM windows").fetchone()[0]
    n_ann_b = con_b.execute("SELECT COUNT(*) FROM annotations").fetchone()[0]
    con_b.close()

    # Use max_win_id + 1 as offset so B IDs never collide with A IDs
    offset = max_win_id + 1

    print(f"\nDB A: {n_windows_a:,} windows, {n_ann_a:,} annotations")
    print(f"DB B: {n_windows_b:,} windows, {n_ann_b:,} annotations")
    print(f"Window ID offset for B: {offset:,}")
    print(f"Output: {args.output}")
    print(f"Expected combined: {n_windows_a + n_windows_b:,} windows, "
          f"{n_ann_a + n_ann_b:,} annotations")

    if args.dry_run:
        print("\nDRY RUN — no changes made.")
        return

    # Merge audio_sources from both DBs
    src_a = get_metadata(db_a, "audio_sources") or {}
    src_b = get_metadata(db_b, "audio_sources") or {}
    globs_a = src_a.get("audio_globs", [])
    globs_b = src_b.get("audio_globs", [])
    combined_sources = {"audio_globs": globs_a + globs_b}

    print("\nCreating output DB...")
    out_db = create_output_db(
        args.output, cfg_a,
        get_metadata(db_a, "model_config"),
        combined_sources
    )

    print("Copying DB A contents...")
    w_a, a_a = copy_db_contents(db_a, out_db, window_id_offset=0)
    print(f"  {w_a:,} windows, {a_a:,} annotations copied from A")

    print("Copying DB B contents...")
    w_b, a_b = copy_db_contents(db_b, out_db, window_id_offset=offset)
    print(f"  {w_b:,} windows, {a_b:,} annotations copied from B")

    print("\nMerging USearch indexes...")
    out_index = os.path.join(args.output, "usearch.index")
    total_vecs = merge_usearch_indexes(idx_a, idx_b, out_index, offset, cfg_a)

    # Final verification
    out_con = sqlite3.connect(out_db)
    total_win = out_con.execute("SELECT COUNT(*) FROM windows").fetchone()[0]
    total_ann = out_con.execute("SELECT COUNT(*) FROM annotations").fetchone()[0]
    pos = dict(out_con.execute(
        "SELECT label, COUNT(*) FROM annotations WHERE label_type=1 GROUP BY label"
    ).fetchall())
    neg = dict(out_con.execute(
        "SELECT label, COUNT(*) FROM annotations WHERE label_type=2 GROUP BY label"
    ).fetchall())
    out_con.close()

    print(f"\n{'='*60}")
    print(f"MERGE COMPLETE")
    print(f"  Windows   : {total_win:,}")
    print(f"  Vectors   : {total_vecs:,}")
    print(f"  Annotations: {total_ann:,}")
    print(f"  Positive  : {pos}")
    print(f"  Negative  : {neg}")
    print(f"  Output    : {args.output}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
