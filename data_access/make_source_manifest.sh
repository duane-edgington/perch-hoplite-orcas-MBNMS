#!/usr/bin/env bash
#
# make_source_manifest.sh — generate SOURCE_MANIFEST.csv from the public bucket.
#
# Lists the raw objects for each month used in this work and writes the manifest
# that lets a re-runner pull exactly our inputs.
#
# Usage: BUCKET=<bucket-name> ./make_source_manifest.sh > SOURCE_MANIFEST.csv
#
set -ue

BUCKET="${BUCKET:?set BUCKET to the public raw-audio bucket name}"

# month:role pairs used in this work
MONTHS="2018/04:training 2018/05:held-out 2020/10:training 2026/04:training 2024/09:exploratory"

echo "month,date,raw_filename,aws_bucket,aws_key,byte_size,source_sample_rate_hz,role,notes"

for entry in ${MONTHS}; do
  path="${entry%%:*}"
  role="${entry##*:}"
  month="${path/\//-}"

  aws s3 ls --no-sign-request --recursive "s3://${BUCKET}/${path}/" \
  | while read -r _date _time size key; do
      fname="$(basename "${key}")"
      # MARS_YYYYMMDD_HHMMSS.wav -> YYYY-MM-DD
      stamp="${fname#MARS_}"; stamp="${stamp%%_*}"
      day="${stamp:0:4}-${stamp:4:2}-${stamp:6:2}"
      echo "${month},${day},${fname},${BUCKET},${key},${size},256000,${role},"
    done
done
