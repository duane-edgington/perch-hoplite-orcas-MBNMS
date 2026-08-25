# REPRODUCE.md — public raw audio to published results

Every step from the public archive to the numbers in the poster. Nothing here requires credentials or access to MBARI systems.

There are five stages. Only stage 2 (resampling) and stage 3 (embedding) are expensive; everything after them takes seconds to minutes.

```
1. Get raw audio        AWS Open Data, free, no credentials
2. Resample             SoX → 32 kHz 16-bit          ~1 day per month
3. Embed                Perch V2 → 5 s vectors       ~40 min per month on a GB10
4. Check coverage       which days actually recorded?      seconds
5. Infer + evaluate     linear probe → detections          minutes
```

---

## Stage 0 — Environment

Two separate virtual environments are used, and mixing them is the most common source of trouble:

| Environment | Used for | Why separate |
|---|---|---|
| `perch-hoplite` venv | training, inference, review, plotting | this repository's `requirements.txt` |
| `perch-pytorch` venv | embedding only (stage 3) | pulls in the PyTorch Perch V2 port and its weights |

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install git+https://github.com/google-research/perch-hoplite.git
```

For stage 3 you additionally need [github.com/duane-edgington/perch-pytorch](https://github.com/duane-edgington/perch-pytorch), a pure-PyTorch reimplementation of Perch V2 verified for numerical parity against the TensorFlow SavedModel. No TensorFlow is required anywhere in this pipeline.

Exact versions used to produce the released results — including the SoX version, which affects output bytes — are pinned in [VERSIONS.md](VERSIONS.md).

---

## Stage 1 — Get the raw audio

The MARS recordings are public on AWS Open Data at no cost and with no credentials. Fetch instructions, bucket paths and the per-file manifest of exactly which recordings underlie this work are in [../data_access/how_to_get_raw_audio.md](../data_access/how_to_get_raw_audio.md) and [../data_access/SOURCE_MANIFEST.csv](../data_access/SOURCE_MANIFEST.csv).

The months used here:

| Month | Role |
|---|---|
| April 2018 | training; sustained multi-day Bigg's presence |
| October 2020 | training; peak humpback season, orca-silent |
| April 2026 | training; hard negatives (humpback false positives) |
| **May 2018** | **permanently held out — never trained on, by any model** |
| September 2024 | exploratory; recording ends 19 Sept (see coverage note below) |

---

## Stage 2 — Resample to 32 kHz

Perch V2 expects 32 kHz, 16-bit input.

```bash
cd resampling/
./resample_sox_32k.sh 2018 5           # note: month WITHOUT leading zero
```

For a long run, background it and keep the log:

```bash
nohup ./resample_sox_32k.sh 2024 9 > logs/nohup_resample_2024_09.out &
```

A full month takes roughly one day. The throttled variant `resample_sox_32k_throttled.sh` caps concurrency (gentler on a shared machine, identical output bytes) and additionally accepts a day range:

```bash
./resample_sox_32k_throttled.sh 2018 5 1 7 8    # May 1–7, 8 parallel jobs
```

The SoX parameters — and in particular why `vol 3` must not be dropped — are documented in [../resampling/README.md](../resampling/README.md).

**Verify you reproduced our inputs.** Hash a few of your resampled outputs and compare against [../data_access/CHECKSUMS.md](../data_access/CHECKSUMS.md). If they match, everything downstream is on solid ground. If they don't, check your SoX version first — output can differ across builds.

Output naming is `MARS_<YYYYMMDD>_<HHMMSS>_resampled_32kHz.wav`; downstream tools match on the `_resampled_32kHz` suffix.

---

## Stage 3 — Embed with Perch V2

```bash
source ~/perch-pytorch/venv/bin/activate      # NOT the perch-hoplite venv

nohup python3 pipeline/phase1_embed_torch.py \
    --audio-dir /path/to/resampled_32kHz/2018/05 \
    --db-dir /path/to/db/MARS_20180501_20180531_32kHz_norm \
    --device cuda --compile \
    > logs/embed_may2018_norm.log 2>&1 &
```

| Flag | Meaning |
|---|---|
| `--device cuda` | GPU; strongly recommended |
| `--compile` | `torch.compile`, roughly 2.5× faster — slow first batch, then accelerates |
| `--hop-size-s` | default 5.0, non-overlapping windows |
| `--batch-size` | default 8 |
| `--date YYYYMMDD` | optional; embed a single day |

Re-running is idempotent — already-embedded files are skipped.

**Per-window peak normalization to 0.25 is applied automatically** inside the embedding adapter. It is not a flag and cannot be switched off, because without it the embeddings are wrong for quiet MARS audio. The `_norm` suffix in the database name is a manual convention marking databases built this way; please keep using it.

**Sanity check.** The tool prints an expected-window count at the start and a `Done. NNNNNN embeddings in <db>` line at the end. Confirm they agree:

| Month | Files | Embeddings |
|---|---|---|
| April 2018 | 4,320 | 518,400 |
| May 2018 | 4,464 | 535,680 |
| October 2020 | 4,504 | 535,278 |
| September 2024 | 2,698 | 323,760 |

One harmless quirk: the auto-generated internal *Dataset name* field may show an incorrect end date. Inference queries the database *path*, so this label has no effect.

---

## Stage 4 — Check data coverage (do not skip)

MARS has outages. A "no orca" result on a day with no recording is meaningless, and this check costs seconds.

```bash
sqlite3 /path/to/db/MARS_20240901_20240930_32kHz_norm/hoplite.sqlite \
  "SELECT substr(filename,6,8) AS day, COUNT(*) FROM recordings GROUP BY day ORDER BY day;"
```

Cross-check against the raw archive listing: the database tells you what was resampled and embedded, the archive tells you what was actually recorded. Both can be incomplete, for different reasons.

This is not hypothetical. September 2024 was resampled and embedded cleanly — 2,698 files, 323,760 embeddings, no errors — and the coverage check revealed that recording *stops on 19 September*, with days 1–18 complete, day 19 partial, and nothing after. A power-connector failure had left the hydrophone on internal battery until it died. A documented three-matriline Bigg's encounter on 27 September 2024 therefore has no acoustic record at all. Had we run inference first and reasoned from the result, the conclusion would have been confidently wrong.

---

## Stage 5 — Inference and evaluation

```bash
for model in v4 v10; do
  python3 pipeline/phase2_classify.py infer \
    --db-dir /path/to/db/MARS_20180501_20180531_32kHz_norm \
    --classifier models/orca_${model}.pt \
    --labels orca_call --logit-threshold 0.0 \
    --output-csv results/MARS_20180501_20180531_${model}_orcaval.csv
done
```

A logit floor of 0.0 is used *for evaluation* so that both models' full score distributions are captured. For interpretation, threshold at **+1.16** or above — see [MODEL_CARD.md](MODEL_CARD.md). Note that the inference CSV is per-label (one row per window per class), so raw row counts at floor 0.0 are inflated by also-ran classes.

**Reproduce the held-out comparison.** With both CSVs written and the May database's confirmed labels in place:

```bash
python3 tools/compare_may_holdout.py
```

This scores both models against the ear-confirmed May orca windows and reports recall at 0.0 / +1.16 / +1.5 / +2.0 plus score distributions. Expected output is summarized in [RESULTS.md](RESULTS.md). Paths are defined as constants at the top of the script — edit them to match your layout.

**Threshold sweep against known regions:**

```bash
python3 tools/score_orca_regions.py --help
```

**Interactive review.** The Gradio labeling and review interface is how every confirmed label in this repository was produced — spectrogram, 5 s audio, 30 s context, click to label:

```bash
python3 pipeline/phase2_classify.py review \
    --db-dir /path/to/db/MARS_20180501_20180531_32kHz_norm \
    --classifier models/orca_v10.pt \
    --target-label orca_call --num-results 25 \
    --classes orca_call,humpback_song,dolphin_call,ship_noise,other,unlabeled \
    --audio-dir /path/to/resampled_32kHz/2018/05 \
    --spectrogram-type mel --colormap viridis \
    --serve --port 7861
```

Two hard-won practical notes: use **Chrome in incognito mode**. A normal window can serve stale cached state and the page will appear to hang while the server is perfectly healthy — try incognito before debugging anything else. Safari fails separately on audio playback with data URIs.

Thirty seconds of context matters more than you would expect. Several April 2026 clips sounded unambiguously like orca within their 5-second window, and the surrounding 30 seconds revealed humpback vocalizations throughout — which is why those candidates are recorded as ambiguous rather than confirmed.

---

## What "confirmed" means here

Every orca label in this repository was listened to by an expert annotator. Detections are candidates; labels are judgments. Where two interpretations could not be separated by ear, the clip was left unlabeled rather than forced into a class — see [DATA.md](DATA.md) for the label schema and annotator provenance, and [RESULTS.md](RESULTS.md) for what remains open.
