# Resampling raw MARS audio to 32 kHz

Perch V2 expects 32 kHz, 16-bit mono input. MARS records at full bandwidth, so every month must be resampled before embedding.

Two scripts, identical SoX parameters, identical output bytes:

| Script | Use |
|---|---|
| `resample_sox_32k.sh` | The script that produced every month in this work. One `sox` process per file. |
| `resample_sox_32k_throttled.sh` | Same parameters, capped concurrency, optional day range. Gentler on a shared machine. |

Concurrency affects speed and system load only, never the output.

```bash
./resample_sox_32k.sh 2018 5                      # whole month
./resample_sox_32k_throttled.sh 2018 5 1 7 8      # May 1–7, 8 parallel jobs
```

Note the month argument takes **no leading zero** — `5`, not `05`. A full month takes roughly a day. Output is written as `MARS_<YYYYMMDD>_<HHMMSS>_resampled_32kHz.wav`; downstream tools match on the `_resampled_32kHz` suffix, so please preserve it.

Both scripts have input and output base directories set at the top for the MBARI environment. Edit those two variables for your own layout.

---

## The command

```
sox <in> -b 16 <out> rate -v 32000 highpass 10 fade 0.1 -0 0.1 vol 3
```

| Flag | What it does | Why |
|---|---|---|
| `rate -v 32000` | Resample to 32 kHz, very-high-quality filter | Perch V2's expected input rate |
| `-b 16` | 16-bit output | Required by the model input spec |
| `highpass 10` | 10 Hz highpass | Removes DC offset |
| `vol 3` | **Voltage calibration** | Converts raw hydrophone output to volts — see below |
| `fade 0.1 -0 0.1` | 0.1 s logarithmic fade in and out, full-duration hold | Avoids edge transients |

---

## About `vol 3`

**It is a calibration step, not an optional gain, and it must not be dropped.**

It converts the raw hydrophone output to volts — the physical unit the science works in — and it is applied to every 32 kHz resample in this project, always, by design. Removing it produces audio that is not in the units the rest of the analysis assumes.

It looks like a gain boost, which invites two reasonable-sounding but wrong objections:

**"Won't 3× cause clipping?"** In practice, no. Signals of genuine interest in this dataset do not approach full scale. SoX may emit `vol clipped` or `dither clipped` warnings on occasional samples; these are not a reason to remove the calibration.

**"Shouldn't the amplitude be left alone for the model?"** The opposite problem is the real one. MARS recordings at 891 m are *very* quiet — typical peak amplitudes of 0.0015 to 0.003. Low amplitude, not clipping, is what this pipeline is built around, and it is handled downstream by per-window peak normalization to 0.25 before Perch V2 (see [../docs/REPRODUCE.md](../docs/REPRODUCE.md) stage 3). That normalization is mandatory and automatic; without it, embeddings diverge badly from the reference implementation.

So: `vol 3` calibrates to volts; normalization handles the quietness. Both, always.

---

## Pin your SoX version

SoX output can differ across versions and builds, which makes the version part of the reproducibility record rather than an incidental detail.

**All released data used SoX 14.4.2.** Check yours with `sox --version`, and verify your output against [../data_access/CHECKSUMS.md](../data_access/CHECKSUMS.md) before spending GPU time on embedding.

---

## Storage

Resampled audio is bulky — roughly 723 GB for the months used here — and is deliberately **not published**, because it is exactly reproducible from public raw audio plus this script at a pinned version. That is what makes deleting it safe, and it is worth doing: process a chunk, extract what supports the analysis, delete the bulk.

Keep the embedding databases and confirmed clips. The intermediate WAVs are regenerable on demand.
