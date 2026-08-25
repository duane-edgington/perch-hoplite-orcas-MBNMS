#!/usr/bin/env python3
"""extract_example_clips.py
Extract 10 peak-normalized 5-second example clips from annotated MARS windows.

Selects 2 clips per class (orca_call, dolphin_call, humpback_song, ship_noise,
other/background) from the April 2018 and October 2020 annotated DBs.

Each clip is:
  - Extracted from the source WAV at the annotated offset
  - Peak-normalized to 0.25 (idempotent with Perch's internal normalization)
  - Saved as 32kHz 16-bit PCM WAV
  - Named: {label}_{date}_{offset_s}s.wav

Output: /mnt/PAM_Analysis/perch-hoplite/example_clips/
"""
import json
import os
import sqlite3
import struct
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PAM_ROOT    = Path("/mnt/PAM_Analysis")
PERCH_ROOT  = PAM_ROOT / "perch-hoplite"
OUTPUT_DIR  = PERCH_ROOT / "example_clips"

DBS = [
    (PERCH_ROOT / "db" / "MARS_20180401_20180430_32kHz_norm" / "hoplite.sqlite",
     PAM_ROOT / "GoogleMultiSpeciesWhaleModel2" / "resampled_32kHz" / "2018" / "04"),
    (PERCH_ROOT / "db" / "MARS_20201001_20201031_32kHz_norm" / "hoplite.sqlite",
     PAM_ROOT / "GoogleMultiSpeciesWhaleModel2" / "resampled_32kHz" / "2020" / "10"),
]

CLASSES = {
    "orca_call":     2,
    "dolphin_call":  2,
    "humpback_song": 2,
    "ship_noise":    2,
    "other":         2,
}
PEAK_TARGET = 0.25
WINDOW_S    = 5.0
SR          = 32000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def peak_normalize(audio: np.ndarray, target: float = PEAK_TARGET) -> np.ndarray:
    """Peak-normalize to target amplitude. Matches perch_hoplite_torch_adapter."""
    audio = audio.astype(np.float64)
    peak  = np.abs(audio).max()
    if peak > 1e-12:
        audio = audio * (target / peak)
    return audio.astype(np.float32)


def get_annotations(db_path: Path, label: str, n: int) -> list:
    """Return up to n annotations for a label from a DB."""
    con = sqlite3.connect(db_path)
    rows = con.execute("""
        SELECT r.filename, a.offsets
        FROM annotations a
        JOIN recordings r ON a.recording_id = r.id
        WHERE a.label = ? AND a.label_type = 1
        ORDER BY RANDOM()
        LIMIT ?
    """, (label, n)).fetchall()
    con.close()
    results = []
    for fname, off_blob in rows:
        if isinstance(off_blob, (bytes, bytearray)) and len(off_blob) >= 16:
            start_s, end_s = struct.unpack_from("<dd", off_blob)
        else:
            start_s, end_s = 0.0, WINDOW_S
        results.append({"filename": fname, "start_s": start_s, "end_s": end_s})
    return results


def extract_clip(audio_dir: Path, filename: str,
                 start_s: float, end_s: float):
    """Load and return a 5-second audio clip."""
    wav_path = audio_dir / filename
    if not wav_path.exists():
        print(f"  WARNING: {wav_path} not found — skipping")
        return None
    start_smp = int(start_s * SR)
    end_smp   = int(end_s   * SR)
    audio, sr_file = sf.read(str(wav_path), start=start_smp, stop=end_smp,
                              dtype="float32", always_2d=False)
    if sr_file != SR:
        print(f"  WARNING: sample rate {sr_file} != {SR} for {filename}")
    # Pad to exactly WINDOW_S if short
    target_len = int(WINDOW_S * SR)
    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)))
    elif len(audio) > target_len:
        audio = audio[:target_len]
    return audio


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    collected = {label: [] for label in CLASSES}

    for db_path, audio_dir in DBS:
        if not db_path.exists():
            print(f"Skipping missing DB: {db_path}")
            continue
        for label, needed in CLASSES.items():
            already = len(collected[label])
            if already >= needed:
                continue
            anns = get_annotations(db_path, label, needed - already)
            for ann in anns:
                audio = extract_clip(audio_dir, ann["filename"],
                                     ann["start_s"], ann["end_s"])
                if audio is None:
                    continue
                collected[label].append({
                    "audio":    audio,
                    "filename": ann["filename"],
                    "start_s":  ann["start_s"],
                    "label":    label,
                })

    print(f"{'Class':<20} {'Source file':<45} {'Offset':>8}  {'Raw peak':>10}  {'Norm peak':>10}")
    print("-" * 100)

    manifest = []
    for label, clips in collected.items():
        if not clips:
            print(f"{label:<20}  *** NO CLIPS FOUND ***")
            continue
        for clip in clips:
            raw_peak   = float(np.abs(clip["audio"]).max())
            norm_audio = peak_normalize(clip["audio"])
            norm_peak  = float(np.abs(norm_audio).max())

            date_str = clip["filename"].split("_")[1]
            offset   = int(clip["start_s"])
            out_name = f"{label}_{date_str}_{offset:04d}s.wav"
            out_path = OUTPUT_DIR / out_name

            sf.write(str(out_path), norm_audio, SR,
                     subtype="PCM_16", format="WAV")

            print(f"{label:<20}  {clip['filename']:<45}  {clip['start_s']:>6.1f}s"
                  f"  {raw_peak:>10.5f}  {norm_peak:>10.5f}")

            manifest.append({
                "output_file": out_name,
                "label":       label,
                "source_file": clip["filename"],
                "offset_s":    clip["start_s"],
                "raw_peak":    round(raw_peak, 6),
                "norm_peak":   round(norm_peak, 6),
            })

    manifest_path = OUTPUT_DIR / "manifest.json"
    with open(str(manifest_path), "w") as f:
        json.dump({
            "description": (
                "10 peak-normalized (0.25) 5-second MARS hydrophone clips, "
                "2 per class. For use as Perch V2 validation/test examples."
            ),
            "sample_rate":  SR,
            "window_s":     WINDOW_S,
            "peak_target":  PEAK_TARGET,
            "normalization": "per-window peak normalization to 0.25 — "
                             "idempotent with Perch internal peak_norm",
            "clips": manifest,
        }, f, indent=2)

    print()
    print(f"Wrote {len(manifest)} clips + manifest to {OUTPUT_DIR}")
    print()
    print("Summary:")
    for label in CLASSES:
        n = len(collected[label])
        status = "OK" if n == CLASSES[label] else f"WARNING: only {n}"
        print(f"  {label:<20} {n}/{CLASSES[label]}  {status}")


if __name__ == "__main__":
    main()
