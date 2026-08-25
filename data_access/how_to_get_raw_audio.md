# Getting the raw audio

The MARS hydrophone recordings underlying this work are **already public, free, and require no credentials**. They are part of the Pacific Ocean Sound archive on the AWS Open Data registry. We do not re-host them.

> **Maintainer note — verify before release.** The bucket names and key layout below are stated from project knowledge and marked where confirmation is needed. Please confirm each against the live registry entry and paste the verified values before this repository goes public. A wrong bucket path is the single most likely reason a first-time re-runner gives up.

---

## Registry entry

Start here for the canonical, current description of the dataset, its license terms, and how to cite it:

- AWS Open Data registry: <https://registry.opendata.aws/pacific-sound/>
- MBARI's Pacific Ocean Sound project page: <https://www.mbari.org/data/passive-acoustic-data/>

Please cite the raw dataset independently of this repository, using whatever citation the registry entry specifies.

---

## What you need

This work uses the **full-bandwidth 256 kHz MARS recordings**, which are then resampled to 32 kHz locally (stage 2 of [../docs/REPRODUCE.md](../docs/REPRODUCE.md)).

Decimated versions of the archive are also published at lower sample rates. **Do not substitute them.** Our resampling chain starts from the full-rate originals, and starting anywhere else will not reproduce our checksums — nor, potentially, our embeddings.

Files follow the naming convention `MARS_<YYYYMMDD>_<HHMMSS>.wav`, organized by year and month.

`TO VERIFY:` exact bucket name and key prefix. Project knowledge indicates a bucket along the lines of `pacific-sound-256khz` with keys of the form `<YYYY>/<MM>/MARS_<YYYYMMDD>_<HHMMSS>.wav`. Confirm against the registry entry above and correct the commands below to match.

---

## Fetching

The bucket is public, so no credentials are needed — but the AWS CLI still requires `--no-sign-request` to skip credential lookup:

```bash
# List what is available for a month
aws s3 ls --no-sign-request s3://<BUCKET>/2018/05/

# Pull one month (this is large — check the listing size first)
aws s3 cp --no-sign-request --recursive \
    s3://<BUCKET>/2018/05/ \
    ./raw/2018/05/

# Pull a single day
aws s3 cp --no-sign-request --recursive \
    --exclude "*" --include "MARS_20180512_*" \
    s3://<BUCKET>/2018/05/ \
    ./raw/2018/05/
```

Without the AWS CLI, the same objects are reachable over plain HTTPS:

```bash
curl -O https://<BUCKET>.s3.amazonaws.com/2018/05/MARS_20180512_083912.wav
```

---

## Which files, exactly

[`SOURCE_MANIFEST.csv`](SOURCE_MANIFEST.csv) lists the specific recordings that underlie the published results, so you can pull exactly our inputs rather than guessing at date ranges.

If you only want to reproduce the headline held-out result, **you need May 2018 alone** — one month, and the comparison in [../docs/RESULTS.md](../docs/RESULTS.md) falls out of it. Reproducing the training pipeline end to end additionally needs April 2018, October 2020, and April 2026.

---

## A word on scale

A month of full-rate MARS audio is substantial, and resampling one takes roughly a day. If you are exploring rather than reproducing, start with a single day — 12 May 2018 is the densest confirmed orca day in the archive, with 181 confirmed calls, and makes a good first target.

Once resampled, verify a few files against [CHECKSUMS.md](CHECKSUMS.md) before spending GPU time on embedding. Catching a mismatch there costs minutes; catching it after inference costs a day.
