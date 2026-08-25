#!/usr/bin/env python3
"""phase1_embed_torch.py
Pure-PyTorch Perch V2 embedding pipeline for MARS hydrophone recordings.

Replaces the Colab-based phase1_embed workflow by running entirely on
spark-ae0e (GB10 DGX) using the native PyTorch Perch V2 implementation
in ~/perch-pytorch. No TensorFlow, no Colab, no Google Drive.

Requirements:
    - ~/perch-pytorch/venv (PyTorch stack, see perch-pytorch README)
    - ~/perch-pytorch/perch_weights/weights.npz
    - ~/perch-pytorch/perch_weights/graph_manifest.json
    - ~/perch-pytorch/const__pad1_output_0.npy  (exact mel reference)
    - ~/perch-pytorch/perch_hoplite_torch_adapter.py
    - perch-hoplite installed in the perch-pytorch venv

Run from ~/perch-hoplite with the perch-pytorch venv active:
    source ~/perch-pytorch/venv/bin/activate

Usage examples:

    # Embed a single day (April 13 2018):
    python3 phase1_embed_torch.py \\
        --audio-dir /mnt/PAM_Analysis/GoogleMultiSpeciesWhaleModel2/resampled_32kHz/2018/04 \\
        --date 20180413 \\
        --db-dir /mnt/PAM_Analysis/perch-hoplite/db/MARS_20180413_torch_32kHz \\
        --device cuda

    # Embed all of April 2018:
    python3 phase1_embed_torch.py \\
        --audio-dir /mnt/PAM_Analysis/GoogleMultiSpeciesWhaleModel2/resampled_32kHz/2018/04 \\
        --db-dir /mnt/PAM_Analysis/perch-hoplite/db/MARS_20180401_20180430_32kHz \\
        --device cuda

    # Embed October 2020:
    python3 phase1_embed_torch.py \\
        --audio-dir /mnt/PAM_Analysis/GoogleMultiSpeciesWhaleModel2/resampled_32kHz/2020/10 \\
        --db-dir /mnt/PAM_Analysis/perch-hoplite/db/MARS_20201001_20201031_32kHz \\
        --device cuda

    # CPU-only (slower, for testing):
    python3 phase1_embed_torch.py \\
        --audio-dir /mnt/PAM_Analysis/GoogleMultiSpeciesWhaleModel2/resampled_32kHz/2018/04 \\
        --date 20180413 \\
        --db-dir /tmp/test_db \\
        --device cpu

Notes:
    - DB format is identical to Colab-generated DBs: same hoplite SQLite
      schema + USearch float16 index. All downstream tools (phase2_classify.py,
      infer, review) work without modification.
    - New DBs created by this script do NOT contain logit_slope/logit_intercept
      in the model_config (those were a Colab notebook artifact), so the
      post-download sqlite3 patch is NOT needed.
    - torch.compile is enabled by default for ~2.5x throughput on GB10.
      Disable with --no-compile for debugging.
    - The script is idempotent: re-running on a partially-embedded DB will
      skip already-embedded files (handle_duplicates='skip').
"""

import argparse
import os
import sys
import time
import logging
import glob
from pathlib import Path

log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────
PERCH_PYTORCH_DIR = Path.home() / "perch-pytorch"
WEIGHTS_DIR       = PERCH_PYTORCH_DIR / "perch_weights"
EXACT_MEL_NPY     = PERCH_PYTORCH_DIR / "const__pad1_output_0.npy"
ADAPTER_SCRIPT    = PERCH_PYTORCH_DIR / "perch_hoplite_torch_adapter.py"

# Default DB base directory
DB_BASE = Path("/mnt/PAM_Analysis/perch-hoplite/db")

# Audio base paths
AUDIO_BASE_32K = Path("/mnt/PAM_Analysis/GoogleMultiSpeciesWhaleModel2/resampled_32kHz")


def check_environment():
    """Verify all required files are present before starting."""
    errors = []
    for p in [WEIGHTS_DIR / "weights.npz",
              WEIGHTS_DIR / "graph_manifest.json",
              EXACT_MEL_NPY,
              ADAPTER_SCRIPT]:
        if not p.exists():
            errors.append(f"  Missing: {p}")
    if errors:
        print("ERROR — required files not found:")
        for e in errors:
            print(e)
        print("\nEnsure ~/perch-pytorch is set up per its README.")
        sys.exit(1)

    # Check PyTorch is importable
    try:
        import torch
        print(f"PyTorch {torch.__version__} — CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("ERROR: torch not importable. Activate ~/perch-pytorch/venv first:")
        print("  source ~/perch-pytorch/venv/bin/activate")
        sys.exit(1)


def build_dataset_name(audio_dir: Path, date: str | None) -> str:
    """Build a dataset name from audio directory and optional date filter."""
    # Infer year/month from path e.g. .../resampled_32kHz/2018/04
    parts = audio_dir.parts
    try:
        yr_idx = next(i for i, p in enumerate(parts) if p.isdigit() and len(p) == 4)
        year  = parts[yr_idx]
        month = parts[yr_idx + 1] if yr_idx + 1 < len(parts) else "XX"
    except StopIteration:
        year, month = "XXXX", "XX"

    if date:
        return f"MARS_{date}_{date}_32kHz"
    else:
        # All files in the directory
        first_day = f"{year}{month}01"
        # Find last day from actual files
        wavs = sorted(glob.glob(str(audio_dir / "*.wav")))
        if wavs:
            import re
            m = re.search(r'MARS_(\d{8})_', Path(wavs[-1]).name)
            last_day = m.group(1) if m else f"{year}{month}30"
        else:
            last_day = f"{year}{month}30"
        return f"MARS_{first_day}_{last_day}_32kHz"


def get_audio_files(audio_dir: Path, date: str | None) -> list[Path]:
    """Return sorted list of WAV files, optionally filtered to a single date."""
    all_wavs = sorted(audio_dir.glob("*.wav"))
    if date:
        filtered = [f for f in all_wavs if date in f.name]
        print(f"Date filter '{date}': {len(filtered)} / {len(all_wavs)} files")
        return filtered
    return all_wavs


def embed_with_adapter(
    audio_files: list[Path],
    db_dir: Path,
    dataset_name: str,
    device: str,
    use_compile: bool,
    hop_size_s: float,
    batch_size: int,
    date: str | None = None,
):
    """Call perch_hoplite_torch_adapter to embed audio files into a hoplite DB."""

    # Add perch-pytorch to sys.path so we can import the adapter
    sys.path.insert(0, str(PERCH_PYTORCH_DIR))
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "perch_hoplite_torch_adapter", str(ADAPTER_SCRIPT))
    adapter_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter_mod)

    db_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = audio_files[0].parent if audio_files else Path(".")

    # If date-filtered, we need to pass files individually.
    # Use adapter's file-list mode if available, otherwise embed folder.
    # Build file glob — if date specified, match only that date's files
    if date:
        file_glob = f"MARS_{date}_*.wav"
    else:
        file_glob = "*.wav"

    log.info("Calling build_db: audio_dir=%s glob=%s db_dir=%s",
             audio_dir, file_glob, db_dir)

    adapter_mod.build_db(
        audio_dir=str(audio_dir),
        file_glob=file_glob,
        db_dir=str(db_dir),
        weights_dir=str(WEIGHTS_DIR),
        exact_mel=str(EXACT_MEL_NPY) if EXACT_MEL_NPY.exists() else None,
        device=device,
        dataset_name=dataset_name,
        batch_size=batch_size,
        hop_size_s=hop_size_s,
        handle_duplicates='skip',
        use_compile=use_compile,
    )


def patch_model_config(db_dir: Path, dataset_name: str, sample_rate: int = 32000):
    """
    Write a clean model_config to hoplite_metadata — without logit_slope/
    logit_intercept (Colab artifact). This script creates DBs via the
    adapter which should already be clean, but we patch defensively.
    """
    import sqlite3, json
    db_path = db_dir / "hoplite.sqlite"
    if not db_path.exists():
        return
    clean_config = json.dumps({
        "model_key": "taxonomy_model_tf",
        "embedding_dim": 1536,
        "model_config": {
            "window_size_s": 5.0,
            "hop_size_s": 5.0,
            "sample_rate": sample_rate,
            "tfhub_path": "google/bird-vocalization-classifier/tensorFlow2/perch_v2",
            "tfhub_version": 2,
            "model_path": ""
        },
        "logits_key": None,
        "logits_idxes": None,
    })
    con = sqlite3.connect(str(db_path))
    existing = con.execute(
        "SELECT value FROM hoplite_metadata WHERE key='model_config'"
    ).fetchone()
    if existing:
        cfg = json.loads(existing[0])
        if "logit_slope" in cfg.get("model_config", {}) or \
           "logit_intercept" in cfg.get("model_config", {}):
            con.execute(
                "UPDATE hoplite_metadata SET value=? WHERE key='model_config'",
                (clean_config,)
            )
            con.commit()
            log.info("Patched model_config: removed logit_slope/logit_intercept")
        else:
            log.info("model_config already clean — no patch needed")
    else:
        con.execute(
            "INSERT INTO hoplite_metadata (key, value) VALUES ('model_config', ?)",
            (clean_config,)
        )
        con.commit()
        log.info("Inserted clean model_config into hoplite_metadata")
    con.close()


def verify_db(db_dir: Path) -> int:
    """Verify the DB and return window count."""
    import sqlite3
    db_path = db_dir / "hoplite.sqlite"
    if not db_path.exists():
        return 0
    con = sqlite3.connect(str(db_path))
    count = con.execute("SELECT COUNT(*) FROM windows").fetchone()[0]
    con.close()
    return count


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--audio-dir", required=True,
        help="Directory containing resampled 32kHz WAV files. "
             "E.g. /mnt/PAM_Analysis/.../resampled_32kHz/2018/04")
    ap.add_argument("--db-dir", required=True,
        help="Output hoplite DB directory (created if needed). "
             "E.g. /mnt/PAM_Analysis/perch-hoplite/db/MARS_20180413_torch_32kHz")
    ap.add_argument("--date", default=None,
        help="Embed only files matching this date string YYYYMMDD. "
             "Omit to embed all files in --audio-dir.")
    ap.add_argument("--device", default="cuda",
        choices=["cuda", "cpu"],
        help="PyTorch device (default: cuda)")
    ap.add_argument("--compile", action="store_true", default=False,
        help="Enable torch.compile (~2.5x faster on GB10; slow first batch)")
    ap.add_argument("--hop-size-s", type=float, default=5.0,
        help="Window hop size in seconds (default 5.0 = non-overlapping)")
    ap.add_argument("--batch-size", type=int, default=8,
        help="Embedding batch size (default 8; increase for more GPU throughput)")
    ap.add_argument("--dataset-name", default=None,
        help="Override auto-generated dataset name")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    print("=" * 60)
    print("Phase 1 Embedding — PyTorch Perch V2 (spark-ae0e GB10)")
    print("=" * 60)

    # Check environment
    check_environment()

    audio_dir  = Path(args.audio_dir)
    db_dir     = Path(args.db_dir)
    use_compile = args.compile

    if not audio_dir.exists():
        print(f"ERROR: audio directory not found: {audio_dir}")
        sys.exit(1)

    # Get file list
    audio_files = get_audio_files(audio_dir, args.date)
    if not audio_files:
        print(f"ERROR: no WAV files found in {audio_dir}"
              + (f" matching date {args.date}" if args.date else ""))
        sys.exit(1)

    # Dataset name
    dataset_name = args.dataset_name or build_dataset_name(audio_dir, args.date)

    print(f"\nConfiguration:")
    print(f"  Audio dir   : {audio_dir}")
    print(f"  Files       : {len(audio_files)}")
    print(f"  DB dir      : {db_dir}")
    print(f"  Dataset     : {dataset_name}")
    print(f"  Device      : {args.device}")
    print(f"  Compile     : {use_compile}")
    print(f"  Hop size    : {args.hop_size_s}s")
    print(f"  Batch size  : {args.batch_size}")
    print(f"  Expected windows: {len(audio_files) * int(600 / args.hop_size_s)}")

    t0 = time.time()

    # Run embedding
    embed_with_adapter(
        audio_files=audio_files,
        db_dir=db_dir,
        dataset_name=dataset_name,
        device=args.device,
        use_compile=use_compile,
        hop_size_s=args.hop_size_s,
        batch_size=args.batch_size,
        date=args.date,
    )

    # Patch model_config defensively
    patch_model_config(db_dir, dataset_name)

    # Verify
    window_count = verify_db(db_dir)
    elapsed = time.time() - t0

    print(f"\n{'=' * 60}")
    print(f"Embedding complete.")
    print(f"  Windows in DB : {window_count}")
    print(f"  Expected      : {len(audio_files) * int(600 / args.hop_size_s)}")
    print(f"  Elapsed       : {elapsed/60:.1f} minutes")
    if elapsed > 0:
        print(f"  Throughput    : {window_count / elapsed:.1f} windows/sec")
    print(f"  DB location   : {db_dir}")
    print(f"\nNext step — run inference (current canonical models are orca_v4.pt / orca_v10.pt;")
    print(f"there is no orca_v4_clean.pt — the _clean models were the retired pre-normalization era):")
    print(f"  python3 phase2_classify.py infer \\")
    print(f"    --db-dir {db_dir} \\")
    print(f"    --classifier /mnt/PAM_Analysis/perch-hoplite/models/orca_v10.pt \\")
    print(f"    --labels orca_call --logit-threshold 0.0 \\")
    print(f"    --output-csv /mnt/PAM_Analysis/perch-hoplite/results/<MONTH>_v10_orcaval.csv")
    print(f"  (repeat with orca_v4.pt -> _v4_orcaval.csv to compare models)")
    print(f"  TIP: check day-by-day coverage before interpreting — MARS has outages:")
    print(f"    sqlite3 {db_dir}/hoplite.sqlite \\")
    print(f"      \"SELECT substr(filename,6,8) day, COUNT(*) FROM recordings GROUP BY day ORDER BY day;\"")
    print("=" * 60)


if __name__ == "__main__":
    main()
