# perch-hoplite-orcas-MBNMS

**Detecting Bigg's killer whales in the MBARI MARS hydrophone archive with frozen Perch V2 embeddings and agile modeling.**

Duane R. Edgington and John P. Ryan — Monterey Bay Aquarium Research Institute (MBARI)

This repository accompanies the IEEE OCEANS 2026 Monterey poster (21–24 September 2026). It contains the code, trained classifiers, labels, and reproducibility bundle needed to regenerate our results from **already-public raw audio**, and to build on them.

---

## What this is

The MARS cabled hydrophone sits at 891 m depth in Monterey Canyon, inside Monterey Bay National Marine Sanctuary, and has been recording near-continuously for a decade. The scientific question is simple to state and hard to answer at archive scale: **when are killer whales acoustically present?**

Our approach uses [Google's Perch V2](https://github.com/google-research/perch-hoplite) as a frozen foundation model. Audio is embedded once into 5-second vectors; a small linear probe is then trained on top of those embeddings through an iterative human-in-the-loop labeling loop ([agile modeling](https://arxiv.org/abs/2505.03071)). Because embedding is the only expensive step, each retrain takes about 30 seconds, and a working detector emerges from hours — not months — of expert listening.

The result is `orca_v10`, a five-class classifier (`orca_call`, `humpback_song`, `dolphin_call`, `ship_noise`, `other`) built from roughly 8–10 hours of expert annotation.

**Headline result, on a month no model ever trained on.** May 2018 is a permanently held-out test month. Against 196 ear-confirmed orca windows there:

| | orca_v4 (prior production) | orca_v10 (this release) |
|---|---|---|
| Mean logit on confirmed orca | 1.646 | **2.613** |
| Recall at the +1.16 operating threshold | 60.7% | **79.6%** |
| Recall at +2.0 | 39.3% | **59.2%** |
| Higher score, head-to-head (195 shared windows) | 3 | **192** |

Reviewing every one of v10's 14 above-threshold detections that were *not* already confirmed found **14/14 real orca and zero false positives** — and among them, orca on **four days v4 had missed entirely** (2, 3, 7 and 29 May). May 2018's confirmed orca days went from four to eight. The better model did not merely score known calls higher; it surfaced biology that was previously invisible.

The same detector is equally informative when it hears nothing. October 2020 has documented visual sightings of Bigg's killer whales but no confirmed orca vocalizations — and `orca_v10`, demonstrably more sensitive, still finds none. Absence measured with a good instrument is real absence, consistent with Bigg's whales hunting silently.

Full numbers, caveats and evidence: **[docs/RESULTS.md](docs/RESULTS.md)**. How the models were built: **[docs/METHOD.md](docs/METHOD.md)**.

---

## Reproducing this work

The raw audio is already public on AWS Open Data — we do not re-host it. What this repository publishes is **reproducibility**: the exact resampling parameters, the pinned tool versions, checksums so you can confirm you regenerated our inputs byte-for-byte, the trained models, and the labels.

```
public raw MARS audio  (AWS Open Data)
        │
        ▼  resampling/resample_sox_32k.sh      SoX → 32 kHz, 16-bit
   resampled WAV
        │
        ▼  pipeline/phase1_embed_torch.py      Perch V2 → 5 s embeddings
   hoplite embedding DB
        │
        ▼  coverage check                      which days actually have data?
        │
        ▼  pipeline/phase2_classify.py infer   linear probe → detections
   detections CSV  →  review / analysis
```

Step-by-step, with every command: **[docs/REPRODUCE.md](docs/REPRODUCE.md)**.

Two details are easy to get wrong and account for most divergence:

- **Per-window peak normalization to 0.25 is mandatory.** MARS audio at depth has typical peak amplitudes of 0.0015–0.003. Without normalizing each window before Perch V2, embeddings diverge from the reference implementation at cosine 0.43–0.94. This is applied automatically inside the embedding adapter; the `_norm` suffix on a database directory is a naming convention marking it.
- **`vol 3` in the resampling step is a voltage calibration, not an optional gain.** It converts raw hydrophone output to volts, the physical unit the science works in, and is applied to every resample in this project. Do not drop it. Clipping is not a practical concern here; low amplitude is, and normalization handles that downstream.

---

## Repository layout

| Path | Contents |
|---|---|
| `resampling/` | SoX resampling scripts and the rationale for every flag |
| `pipeline/` | Embedding (`phase1_embed_torch.py`) and the main CLI (`phase2_classify.py`: train, infer, review, stats) |
| `tools/` | Held-out evaluation, threshold sweeps, label export, plotting |
| `models/` | Trained classifiers `orca_v4.pt`, `orca_v10.pt` + metrics + [model card](docs/MODEL_CARD.md) |
| `labels/` | Confirmed annotations per month, with schema and provenance |
| `data_access/` | How to fetch the public raw audio, source manifest, checksums |
| `docs/` | Method narrative, results, reproduction guide, pinned versions |

Deliberately **not** here: full embedding databases (~45 GB) and bulk resampled audio (~723 GB), both regenerable from the scripts and manifest; the working repository's exploratory branches and internal notes; and any label not confirmed by ear. If you want something that isn't here, please open an issue — we would rather answer a specific question than publish an archive nobody can navigate.

---

## Quick start

```bash
git clone https://github.com/duane-edgington/perch-hoplite-orcas-MBNMS.git
cd perch-hoplite-orcas-MBNMS
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Inspect the released classifier without any audio:
python3 -c "import json; print(json.dumps(json.load(open('models/orca_v10.metrics.json')), indent=2))"

# Run inference over an existing embedding database:
python3 pipeline/phase2_classify.py infer \
    --db-dir /path/to/MARS_20180501_20180531_32kHz_norm \
    --classifier models/orca_v10.pt \
    --labels orca_call --logit-threshold 1.16 \
    --output-csv may2018_v10_orca.csv
```

Building a database from scratch needs a GPU and the pure-PyTorch Perch V2 port at
[github.com/duane-edgington/perch-pytorch](https://github.com/duane-edgington/perch-pytorch). See [docs/REPRODUCE.md](docs/REPRODUCE.md).

---

## Reading the output

Detection scores are **logits, not probabilities**. The default floor of 0.0 is far too permissive — on months confirmed to be orca-silent it yields hundreds of false positives, which collapse to single digits once thresholded.

**Use +1.16** (the F1-optimal operating threshold) as the primary cutoff, +1.5 for a conservative read. And note that a single global threshold cannot serve all five classes: per-class optima span roughly +0.2 to +2.5. Per-class thresholds are in [docs/MODEL_CARD.md](docs/MODEL_CARD.md).

One more habit worth adopting: **check day-by-day data coverage before interpreting any month.** MARS has had outages. A power-connector failure ended recording on 19 September 2024, which is why a well-documented three-matriline Bigg's encounter on 27 September that year is simply not in the acoustic record. "No orca detected" on a day with no data means nothing at all. The coverage query is in [docs/REPRODUCE.md](docs/REPRODUCE.md).

---

## Citation

If you use this work, please cite the poster and this repository (see `CITATION.cff`). The archived, DOI'd snapshot of models, labels and example clips lives on Zenodo:

> DOI: *to be minted at release — see docs/DATA.md*

Raw audio should be cited independently as the Pacific Ocean Sound / MBARI MARS dataset on AWS Open Data; see [data_access/how_to_get_raw_audio.md](data_access/how_to_get_raw_audio.md).

## License

Apache License 2.0 — code, models, and released label data alike. See `LICENSE` and `NOTICE`.

The raw MARS audio is not ours to relicense; it is distributed under its own terms via AWS Open Data. Perch V2 is Google Research's work under its own license — see `NOTICE`.

## Acknowledgements

Built on [google-research/perch-hoplite](https://github.com/google-research/perch-hoplite). Visual ground truth for killer whale presence comes from the [California Killer Whale Project](https://www.californiakillerwhaleproject.org/). MARS is operated by MBARI.
