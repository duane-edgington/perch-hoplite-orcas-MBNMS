#!/usr/bin/env bash
#
# make_checksums.sh — hash a representative sample of resampled outputs.
#
# Pick a handful of files spanning every month used in the work, hash them,
# and paste the result into CHECKSUMS.md.
#
# Usage: RESAMPLED_ROOT=/path/to/resampled_32kHz ./make_checksums.sh
#
set -ue

ROOT="${RESAMPLED_ROOT:?set RESAMPLED_ROOT to the resampled_32kHz directory}"

echo "# SoX version used:"
sox --version 2>&1 | sed 's/^/#   /'
echo "#"

# One file per month used in this work: first file of a representative day.
for spec in 2018/04:20180413 2018/05:20180512 2020/10:20201005 2026/04:20260421 2024/09:20240905; do
  path="${spec%%:*}"
  day="${spec##*:}"
  f=$(ls "${ROOT}/${path}/MARS_${day}_"*_resampled_32kHz.wav 2>/dev/null | head -1 || true)
  if [ -n "${f}" ]; then
    sha256sum "${f}"
  else
    echo "# MISSING: ${ROOT}/${path}/MARS_${day}_*"
  fi
done
