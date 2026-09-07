# CHECKSUMS.md — verify you reproduced our inputs

The resampling script alone reproduces the *method*. These checksums let you confirm you reproduced the *bytes* — that your SoX build, flags, and source files produced output identical to ours, rather than silently diverging in a way that only shows up as slightly different embeddings later.

**Workflow:** pull the raw audio, run `resampling/resample_sox_32k.sh`, hash a few outputs, compare. Do this before spending GPU time on embedding.

Regenerate or extend the table with [`make_checksums.sh`](make_checksums.sh):

```bash
RESAMPLED_ROOT=/path/to/resampled_32kHz ./make_checksums.sh
```

---

## Reference environment

| | |
|---|---|
| SoX version | 14.4.2 (`/usr/bin/sox`) |
| Flags | `-b 16 rate -v 32000 highpass 10 fade 0.1 -0 0.1 vol 3` |
| Host OS | Ubuntu 24.04.3 LTS |

If your hashes differ, check the SoX version first — that is the most common cause.

---

## Representative resampled outputs

Spanning all months used in this work.

One file per month used in this work. Each is the output of `resample_sox_32k.sh` applied to
the correspondingly named raw recording (`MARS_<stamp>.wav`) from the public bucket for that year.

| Resampled file | sha256 |
|---|---|
| `MARS_20180413_000913_resampled_32kHz.wav` | `d6732a6bbe46b4e7070942ec593831a298de75fdfae2697d1e33dffbbe2345bc` |
| `MARS_20180512_000913_resampled_32kHz.wav` | `86bebcc81e6324ce79ced5927bf43101dc872f234ba31ad5de8e4762d90e83ec` |
| `MARS_20201005_000001_resampled_32kHz.wav` | `a3082f7b210a4cf27b0b1012182125793a99c05cba7c0fad6561c7fe0287ee3e` |
| `MARS_20260421_000000_resampled_32kHz.wav` | `2bb394053e4c42ad72947c17b38e42a580b653f94a00f48eb507f2164e09b520` |
| `MARS_20240905_000756_resampled_32kHz.wav` | `d2f8ae738f4735f1752640565a4cebfadd7ea484b91db1ee3f90c603dcde058e` |

Note the varying start times — `000913`, `000001`, `000756`. Recording start offsets differ by
deployment period, so construct nothing: take filenames from a bucket listing. See
[how_to_get_raw_audio.md](how_to_get_raw_audio.md).

---

## If they don't match

A mismatch does not necessarily mean your data is unusable — SoX resampling differences between builds are typically small. It does mean you are not byte-identical to us, so any downstream difference in results cannot be cleanly attributed.

Worth checking, in order: SoX version and build flags; that you started from the full-rate raw audio rather than a pre-decimated version; that `vol 3` was applied; and that the source file itself matches the byte size in `SOURCE_MANIFEST.csv`.

If you have checked all of that and still differ, please open an issue — we would like to know.
