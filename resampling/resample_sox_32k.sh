#!/usr/bin/env bash
#
# resample_sox_32k.sh — resample raw MARS archive audio to 32 kHz for Perch V2.
#
# This is the script that produced every resampled month used in this work.
# It launches one `sox` process per file. Concurrency affects speed and system
# load ONLY -- not the output bytes. A throttled variant with identical SoX
# parameters is provided as resample_sox_32k_throttled.sh.
#
# Usage:   ./resample_sox_32k.sh <year> <month>
# Example: ./resample_sox_32k.sh 2018 5
#
# NOTE on the month argument: pass it WITHOUT a leading zero (`5`, not `05`).
# The script zero-pads internally when building paths.
#
# Exact invocation used for the September 2024 run:
#   nohup ./resample_sox_32k.sh 2024 9 > logs/nohup_resample_2024_09.out &
#
# A full month takes roughly one day on the reference host.
#
# SoX version used for all released data: SoX v14.4.2 (/usr/bin/sox).
# SoX output can differ across versions and builds, so the version is part of
# the reproducibility record. See ../docs/VERSIONS.md.
#
set -ue

year=$1
month=$2
days=$(seq 1 31)  # 1-31 regardless of month; non-existent days simply no-op

audio_base_dir="/mnt/PAM_Archive"
decimated_base_dir="/mnt/PAM_Analysis/GoogleMultiSpeciesWhaleModel2/resampled_32kHz"

days_line="$(echo "${days}" | tr '\n' ' ')"
in_dir=$(printf "%s/%04d/%02d" ${audio_base_dir} "${year}" "${month}")
out_dir=$(printf "%s/%04d/%02d" ${decimated_base_dir} "${year}" "${month}")
mkdir -p "${out_dir}"

printf "Starting resample_sox_32k.sh: %04d-%02d days: %s\n" "${year}" "${month}" "${days_line}"

# SoX resample directly:
#   rate -v 32000   convert to 32 kHz, -v = very high quality
#   -b 16           16-bit depth (required by the Perch V2 input spec)
#   highpass 10     remove DC offset (10 Hz highpass)
#   vol 3           calibration to volts -- NOT an optional gain (see README.md)
#   fade 0.1 -0 0.1 logarithmic 0.1 s fade in, full-duration hold (-0), 0.1 s fade out
for day in ${days}; do
  prefix=$(printf "%s/MARS_%04d%02d%02d" "${in_dir}" "${year}" "${month}" "${day}")
  for infile in "${prefix}"_*.wav; do
    basename=$(basename "${infile}" .wav)
    outfile="${out_dir}/${basename}_resampled_32kHz.wav"
    echo "infile = ${infile}"
    echo "outfile = ${outfile}"
    sox "${infile}" -b 16 "${outfile}" rate -v 32000 highpass 10 fade 0.1 -0 0.1 vol 3 &
  done
done
wait
