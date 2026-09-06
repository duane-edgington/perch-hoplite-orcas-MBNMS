# Getting the raw audio

The MARS hydrophone recordings underlying this work are **already public, free, and require no credentials**. They are published as part of the Pacific Ocean Sound archive on the AWS Open Data registry. We do not re-host them.

- AWS Open Data registry: <https://registry.opendata.aws/pacific-sound/>
- MBARI Pacific Ocean Sound: <https://www.mbari.org/data/passive-acoustic-data/>

Please cite the raw dataset independently of this repository, using the citation given on the registry page.

---

## Bucket layout

The full-bandwidth recordings are split into **one bucket per year**, in AWS region `us-west-2`:

```
s3://pacific-sound-256khz-<YYYY>/<MM>/MARS_<YYYYMMDD>_<HHMMSS>.wav
```

So May 2018 is `s3://pacific-sound-256khz-2018/05/`. The month is a two-digit prefix; files sit flat inside it. Note the year appears in the *bucket name*, not in the key — `s3://pacific-sound-256khz/2018/05/` does not exist.

Every year this work uses is available: 2016, 2018, 2020, 2024, and 2026.

Decimated versions of the archive are published separately at lower sample rates. **Do not substitute them.** Our resampling chain starts from the full-rate 256 kHz originals, and starting anywhere else will not reproduce our checksums.

## File characteristics

Individual recordings are **10 minutes long**, 256 kHz, mono, 24-bit — 460,800,356 bytes each. Stage 2 resampling reduces both the rate and the bit depth (32 kHz, 16-bit) for Perch V2.

Recordings are not aligned to the hour. May 2018 files begin at `00:09:12`, `00:19:12`, `00:29:12` and so on, and the offset varies by deployment period. **List the bucket rather than constructing filenames** — a generated filename will usually be wrong.

Ten minutes per file means 120 non-overlapping 5-second windows per recording, which is the arithmetic behind the embedding counts in [../docs/REPRODUCE.md](../docs/REPRODUCE.md): May 2018's 4,464 files x 120 = 535,680 embeddings.

---

## Fetching

The buckets are public, so no credentials are needed — but the AWS CLI still requires `--no-sign-request` to skip credential lookup.

```bash
# See what a month holds
aws s3 ls --no-sign-request s3://pacific-sound-256khz-2018/05/

# Pull a single day (~144 files, ~66 GB) -- start here
aws s3 cp --no-sign-request --recursive \
    --exclude "*" --include "MARS_20180512_*" \
    s3://pacific-sound-256khz-2018/05/ \
    ./raw/2018/05/

# Pull a whole month (~4,400 files, ~2 TB) -- check your disk first
aws s3 cp --no-sign-request --recursive \
    s3://pacific-sound-256khz-2018/05/ \
    ./raw/2018/05/
```

Without the AWS CLI, the same objects are reachable over plain HTTPS:

```bash
curl -O https://pacific-sound-256khz-2018.s3.amazonaws.com/05/MARS_20180512_083912.wav
```

---

## Which files, exactly

[`SOURCE_MANIFEST.csv`](SOURCE_MANIFEST.csv) lists the specific recordings underlying the published results, with bucket, key, and byte size — so you can pull exactly our inputs rather than inferring date ranges. Regenerate or extend it with [`make_source_manifest.sh`](make_source_manifest.sh).

The months used, and their buckets:

| Month | Bucket and prefix | Role |
|---|---|---|
| April 2018 | `pacific-sound-256khz-2018/04/` | training |
| **May 2018** | `pacific-sound-256khz-2018/05/` | **permanently held out** |
| October 2020 | `pacific-sound-256khz-2020/10/` | training |
| April 2026 | `pacific-sound-256khz-2026/04/` | training |
| September 2024 | `pacific-sound-256khz-2024/09/` | exploratory (recording ends 19 Sept) |

To reproduce the headline held-out result you need **May 2018 alone**. Reproducing the training pipeline end to end additionally needs April 2018, October 2020, and April 2026.

---

## A word on scale

A month of full-rate audio is roughly 2 TB, and resampling one takes about a day. If you are exploring rather than reproducing, start with a single day — **12 May 2018** is the densest confirmed orca day in the archive, with 181 confirmed calls, and makes a good first target at about 66 GB.

Once resampled, verify a few files against [CHECKSUMS.md](CHECKSUMS.md) before spending GPU time on embedding. Catching a mismatch there costs minutes; catching it after inference costs a day.
