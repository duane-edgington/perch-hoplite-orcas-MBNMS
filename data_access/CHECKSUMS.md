# CHECKSUMS.md — verify you reproduced our inputs

The resampling script alone reproduces the *method*. These checksums let you confirm you reproduced the *bytes* — that your SoX build, flags, and source files produced output identical to ours, rather than silently diverging in a way that only shows up as slightly different embeddings later.

**Workflow:** pull the raw audio, run `resampling/resample_sox_32k.sh`, hash a few outputs, compare. Do this before spending GPU time on embedding.

> **Maintainer note — generate before release.** Run `make_checksums.sh` on the production host and paste its output into the table below. An empty checksums file undermines the whole reproducibility claim.

```bash
sha256sum MARS_20180413_*.wav MARS_20180512_083912_resampled_32kHz.wav ...
```

---

## Reference environment

| | |
|---|---|
| SoX version | 14.4.2 |
| Flags | `-b 16 rate -v 32000 highpass 10 fade 0.1 -0 0.1 vol 3` |
| Host OS | `TO VERIFY` |

If your hashes differ, check the SoX version first — that is the most common cause.

---

## Representative resampled outputs

Spanning all months used in this work.

| Resampled file | sha256 |
|---|---|
| `MARS_20180413_..._resampled_32kHz.wav` | `TO GENERATE` |
| `MARS_20180512_..._resampled_32kHz.wav` | `TO GENERATE` |
| `MARS_20201005_..._resampled_32kHz.wav` | `TO GENERATE` |
| `MARS_20260421_..._resampled_32kHz.wav` | `TO GENERATE` |
| `MARS_20240905_..._resampled_32kHz.wav` | `TO GENERATE` |

---

## If they don't match

A mismatch does not necessarily mean your data is unusable — SoX resampling differences between builds are typically small. It does mean you are not byte-identical to us, so any downstream difference in results cannot be cleanly attributed.

Worth checking, in order: SoX version and build flags; that you started from the full-rate raw audio rather than a pre-decimated version; that `vol 3` was applied; and that the source file itself matches the byte size in `SOURCE_MANIFEST.csv`.

If you have checked all of that and still differ, please open an issue — we would like to know.
