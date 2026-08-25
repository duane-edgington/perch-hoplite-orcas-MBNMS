"""src/infer.py
Inference logic for the Perch-Hoplite pipeline.

Runs a trained LinearClassifier over all embeddings in a DB and writes
detections to a CSV file. Writes to /tmp first then copies to the final
path to avoid ThreadPoolExecutor NFS hang in write_inference_csv.
"""
import csv
import logging
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path

log = logging.getLogger(__name__)


def run_inference(
    db,
    linear_classifier,
    output_csv: str,
    logit_threshold: float = 0.0,
    labels=None,
    classifier_mod=None,
) -> dict:
    """Run inference and write detections to CSV.

    Writes to /tmp first, then copies to output_csv. This avoids the
    ThreadPoolExecutor NFS hang in perch-hoplite's write_inference_csv.

    Parameters
    ----------
    db : HopliteDBInterface
        An open Hoplite database.
    linear_classifier : LinearClassifier
        Trained classifier from perch-hoplite.
    output_csv : str
        Final output path (may be on NFS).
    logit_threshold : float
        Only write detections with logit > threshold.
    labels : list | None
        Labels to include. None = all classifier classes.
    classifier_mod : module | None
        perch_hoplite.agile.classifier module. If None, imported here.

    Returns
    -------
    dict with keys: detection_count, by_label (Counter), elapsed_s
    """
    import time

    if classifier_mod is None:
        from perch_hoplite.agile import classifier as classifier_mod

    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Delete existing file — perch-hoplite's csv_worker_fn appends,
    # so a stale file causes row multiplication
    if out_path.exists():
        out_path.unlink()
        log.info("Removed existing output CSV (append-mode safeguard)")

    t0 = time.monotonic()

    tmp_csv = tempfile.mktemp(suffix=".csv", dir="/tmp", prefix="perch_infer_")
    log.info("Writing inference CSV to local tmp: %s", tmp_csv)

    classifier_mod.write_inference_csv(
        linear_classifier, db, tmp_csv,
        logit_threshold,
        labels=labels,
    )

    log.info("Copying CSV to output: %s", out_path)
    shutil.copy(tmp_csv, str(out_path))
    os.unlink(tmp_csv)

    elapsed = time.monotonic() - t0

    # Count detections
    detection_count = 0
    by_label = Counter()
    try:
        with open(str(out_path), newline="") as f:
            for row in csv.DictReader(f):
                detection_count += 1
                by_label[row.get("label", "?")] += 1
    except Exception:
        pass

    log.info("Inference complete in %.1fs — %d detections", elapsed, detection_count)
    for lbl, cnt in sorted(by_label.items()):
        log.info("  %-40s  %d", lbl, cnt)

    return {
        "detection_count": detection_count,
        "by_label": by_label,
        "elapsed_s": elapsed,
    }
