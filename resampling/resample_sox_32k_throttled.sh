#!/usr/bin/env bash
#
# Resample PAM audio to 32 kHz (by default) with SoX -- throttled variant.
#
# Identical SoX parameters to resample_sox_32k.sh (rate -v, -b 16, highpass 10,
# fade 0.1 -0 0.1, vol 3), so the OUTPUT BYTES ARE THE SAME. The only difference
# is scheduling: this variant caps how many SoX processes run at once instead of
# launching every file into the background simultaneously, so the job behaves the
# same whether run locally or over a remote connection. Files are grouped by day
# for readable progress output.
#
# The `vol 3` step is a voltage calibration, not an optional gain -- see README.md.
# It is applied to every 32 kHz resample in this project, always, by design.
#
# Sample rate:
#   Defaults to 32000 Hz. Override with the SAMPLE_RATE environment variable,
#   e.g.  SAMPLE_RATE=16000 ./resample_sox_32k_throttled.sh 2018 5
#   The output directory leaf and filename suffix follow the rate, so e.g.
#   16000 -> resampled_16kHz/..._resampled_16kHz.wav.
#
# Required args: year month
# Optional args: start_day end_day max_jobs
#
# Examples:
#   ./resample_sox_32k_throttled.sh 2018 5            # whole month, auto max_jobs
#   ./resample_sox_32k_throttled.sh 2018 5 2 2        # only May 2, 2018
#   ./resample_sox_32k_throttled.sh 2018 5 1 7        # May 1-7, 2018
#   ./resample_sox_32k_throttled.sh 2018 5 1 31 8     # whole month, 8 parallel jobs
#   SAMPLE_RATE=16000 ./resample_sox_32k_throttled.sh 2018 5   # 16 kHz instead
#
# Note: set -e is intentionally NOT used here. Managing background jobs with
# -e is fragile, and a single failed conversion should not kill the batch;
# failures are logged and reported at the end instead.

set -u

sample_rate="${SAMPLE_RATE:-32000}"

year=$1
month=$2
start_day="${3:-1}"
end_day="${4:-31}"
max_jobs="${5:-$(nproc 2>/dev/null || echo 4)}"

# Safety: never allow a nonsensical concurrency of < 1 (would deadlock).
[ "${max_jobs}" -ge 1 ] 2>/dev/null || max_jobs=1

# Human-friendly rate label used in paths/filenames, e.g. 32000 -> "32kHz",
# 44100 -> "44.1kHz". Falls back to "<rate>Hz" if awk isn't available.
rate_label=$(awk -v r="${sample_rate}" 'BEGIN{printf "%gkHz", r/1000}' 2>/dev/null)
[ -n "${rate_label}" ] || rate_label="${sample_rate}Hz"

audio_base_dir="/mnt/PAM_Archive"
decimated_base_dir="/mnt/PAM_Analysis/GoogleMultiSpeciesWhaleModel2/resampled_${rate_label}"

in_dir=$(printf "%s/%04d/%02d" "${audio_base_dir}" "${year}" "${month}")
out_dir=$(printf "%s/%04d/%02d" "${decimated_base_dir}" "${year}" "${month}")
mkdir -p "${out_dir}"

# Collect the names of any files that fail to convert, so we can report them.
fail_log=$(mktemp)
trap 'rm -f "${fail_log}"' EXIT

# Block until fewer than max_jobs background jobs are running.
# Uses `wait -n` (event-driven) when available, otherwise falls back to a
# short polling sleep so this works on older bash too.
throttle() {
  while [ "$(jobs -rp | wc -l)" -ge "${max_jobs}" ]; do
    wait -n 2>/dev/null || sleep 0.2
  done
}

# Convert a single file. On failure, record it rather than aborting the run.
# Effects: resample to target rate, highpass 10 Hz (remove DC offset),
# short log fades in/out. `vol` adjustment in this variant.
convert_one() {
  local infile="$1" outfile="$2"
  if ! sox "${infile}" -b 16 "${outfile}" \
        rate -v "${sample_rate}" highpass 10 fade 0.1 -0 0.1 vol 3; then
    echo "${infile}" >> "${fail_log}"
  fi
}

printf "Starting (vol 3): %04d-%02d days %d-%d, rate %s, up to %d parallel job(s)\n" \
  "${year}" "${month}" "${start_day}" "${end_day}" "${rate_label}" "${max_jobs}"

total=0
for day in $(seq "${start_day}" "${end_day}"); do
  prefix=$(printf "%s/MARS_%04d%02d%02d" "${in_dir}" "${year}" "${month}" "${day}")

  # Gather this day's files; nullglob makes a no-match expand to nothing,
  # so days that don't exist (e.g. day 31 in a 30-day month) are skipped.
  shopt -s nullglob
  files=( "${prefix}"_*.wav )
  shopt -u nullglob
  [ "${#files[@]}" -eq 0 ] && continue

  printf "  Day %02d: %d file(s)\n" "${day}" "${#files[@]}"
  for infile in "${files[@]}"; do
    basename=$(basename "${infile}" .wav)
    outfile="${out_dir}/${basename}_resampled_${rate_label}.wav"
    throttle                       # wait for a free slot
    convert_one "${infile}" "${outfile}" &
    total=$((total + 1))
  done

  # To force each day to fully finish before the next begins (strict
  # one-day-at-a-time batching), uncomment the line below. Leaving it
  # commented keeps the pipeline full across day boundaries for best speed.
  # wait
done

wait  # let the final batch of jobs finish

fail_count=$(wc -l < "${fail_log}" | tr -d ' ')
printf "Finished: %d file(s) submitted, %d failure(s)\n" "${total}" "${fail_count}"

if [ "${total}" -eq 0 ]; then
  echo "WARNING: no matching files were found. Check that ${in_dir} is mounted/accessible."
fi
if [ "${fail_count}" -gt 0 ]; then
  echo "Failed files:"
  cat "${fail_log}"
fi
