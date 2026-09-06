#!/usr/bin/env bash
#
# make_source_manifest.sh — generate SOURCE_MANIFEST.csv from the public buckets.
#
# Lists the raw objects for each month used in this work and writes the manifest
# that lets a re-runner pull exactly our inputs.
#
# The Pacific Ocean Sound archive uses ONE BUCKET PER YEAR:
#     s3://pacific-sound-256khz-<YYYY>/<MM>/MARS_<YYYYMMDD>_<HHMMSS>.wav
# The year is in the bucket name, not the key.
#
# Usage:  ./make_source_manifest.sh > SOURCE_MANIFEST.csv
#
# Requires the AWS CLI. No credentials needed (--no-sign-request).
# Takes a few minutes: it lists several thousand objects per month.
#
set -ue

BUCKET_PREFIX="${BUCKET_PREFIX:-pacific-sound-256khz}"

# year/month:role for every month used in this work
MONTHS="2018/04:training 2018/05:held-out 2020/10:training 2026/04:training 2024/09:exploratory"

echo "month,date,raw_filename,aws_bucket,aws_key,byte_size,source_sample_rate_hz,role"

for entry in ${MONTHS}; do
  path="${entry%%:*}"
  role="${entry##*:}"
  year="${path%%/*}"
  mon="${path##*/}"
  bucket="${BUCKET_PREFIX}-${year}"

  aws s3 ls --no-sign-request "s3://${bucket}/${mon}/" \
  | while read -r _date _time size fname; do
      # skip directory lines and blanks
      [ -n "${fname:-}" ] || continue
      case "${fname}" in *.wav) ;; *) continue ;; esac
      # MARS_YYYYMMDD_HHMMSS.wav -> YYYY-MM-DD
      stamp="${fname#MARS_}"; stamp="${stamp%%_*}"
      day="${stamp:0:4}-${stamp:4:2}-${stamp:6:2}"
      echo "${year}-${mon},${day},${fname},${bucket},${mon}/${fname},${size},256000,${role}"
    done
done
