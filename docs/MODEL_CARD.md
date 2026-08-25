# Model card — orca_v10 (and orca_v4)

## What it is

A **linear probe** trained on frozen [Google Perch V2](https://github.com/google-research/perch-hoplite) audio embeddings. Perch V2 is not fine-tuned; it is used purely as a fixed feature extractor. The released artifact is therefore small — a few hundred kilobytes — and trivial to run.

- **Input:** 5-second, 32 kHz, 16-bit mono audio windows, peak-normalized to 0.25, embedded by Perch V2.
- **Output:** five class logits — `orca_call`, `humpback_song`, `dolphin_call`, `ship_noise`, `other`.
- **Files:** `models/orca_v10.pt` (current best), `models/orca_v4.pt` (prior production), `models/orca_v10.metrics.json`.

`orca_v4` is included because it is the model behind most published figures and because it makes the held-out comparison in [RESULTS.md](RESULTS.md) reproducible. For new work, **use `orca_v10`.**

---

## Intended use

Detecting killer whale vocalizations — and the contrastive classes above — in passive acoustic monitoring audio resembling MARS: deep-water cabled hydrophone, Monterey Bay, 32 kHz. Research use.

Reasonable applications include screening large archives for candidate orca activity, generating candidates for expert review, and as a baseline or starting point for other PAM deployments.

**Not intended for**: automated decisions without expert review, real-time alerting, species inventory without ground truth, regulatory or compliance use, or ecotype determination.

---

## Training data

Three months of MARS recordings, deliberately chosen for contrast:

| Month | Contributes |
|---|---|
| April 2018 | sustained Bigg's presence; orca, dolphin, ship, other |
| October 2020 | peak humpback season; orca-silent — the specificity anchor |
| April 2026 | hard negatives — high-scoring humpback misclassified as orca |

**1,076 annotations**, every one confirmed by expert listening. Held-out evaluation set: 459 examples. Per-class support is in `orca_v10.metrics.json` and [RESULTS.md](RESULTS.md).

**May 2018 is permanently excluded from training** and serves as the held-out test month. This is a standing policy, not a one-off split.

---

## Performance

Per-class F1 at F1-optimal thresholds, on the held-out evaluation split:

| Class | F1 | n |
|---|---|---|
| `orca_call` | 0.945 | 61 |
| `ship_noise` | 0.800 | 10 |
| `dolphin_call` | 0.687 | 44 |
| `humpback_song` | 0.619 | 67 |
| `other` | 0.591 | 22 |

Aggregate: ROC-AUC 0.9372, cmap 0.678, on a 459-example evaluation set.

On the **held-out May 2018 month**: 79.6% recall at the operating threshold (v4: 60.7%), and 14/14 precision with zero false positives among above-threshold detections reviewed by ear.

These are **MARS and Monterey Bay numbers.** Do not assume they transfer.

---

## Thresholds — read this before using the model

Scores are **logits, not probabilities.** The inference default floor of 0.0 is uniformly too permissive: on months confirmed orca-silent it produces hundreds of false positives that collapse to single digits under thresholding.

**Primary operating threshold: +1.16.** Conservative: +1.5.

A single global threshold cannot serve all five classes — per-class F1-optimal thresholds span roughly +0.2 to +2.5, and using one value for all of them will silently mis-tune four of the five. Per-class thresholds are recorded in `orca_v10.metrics.json`; take them from there rather than from any number quoted in prose.

---

## Limitations

**Ecotype.** The orca detected here are predominantly Bigg's (transients), largely associated with the CA140 matriline. The model is not validated on Residents or Offshores, and provides **no ecotype classification.**

**Humpback confusion is the dominant failure mode.** Humpback whales have a large and varied repertoire and can produce orca-like sounds. Above-threshold false positives, where they occur, are overwhelmingly humpback. April 2026 produced the sharpest case: 13 high-scoring candidates that sounded like orca in isolation and revealed humpback vocalization throughout their 30-second context. **Those candidates are ambiguous and are not part of any confirmed count in this release.**

Practical consequence: **listen to 30 seconds of context, not just the 5-second window.** The context routinely changes the call.

**`humpback_song` is a misnomer.** The class covers humpback vocalization generally, mixing true song with non-song social calls. That heterogeneity is the leading explanation for its comparatively weak F1. When reporting on a clip, say "humpback vocalization" unless it is plausibly part of a song sequence.

**`other` is a heterogeneous catch-all** and is currently the weakest class. Treat `other` detections as "something is here, unclassified" rather than as a meaningful category.

**`ship_noise` correlates positively with orca presence in this dataset.** Whale-watch and research vessels arrive once orcas are sighted. Any false-positive suppression keying on vessel noise would be exactly wrong on event days.

**Precision is characterized at and above threshold on confirmed sets**, not as an exhaustive month-wide false-alarm rate.

**Single deployment.** One hydrophone, 891 m depth, one bay, four months of labeled data. Transfer to other sites, depths and equipment is plausible and untested.

**Aggregate metrics are not comparable across versions** in this project — evaluation sets differ in size and difficulty between v4 and v10. Compare per-class F1, or compare on the held-out month.

---

## Ethical and practical considerations

Detections are candidates, not conclusions. Every scientific claim in this release rests on expert listening, and we recommend that anyone building on it preserve that step rather than treating model output as ground truth.

A false negative on a rare species can matter more than a false positive, depending on the application — the +1.16 threshold is tuned for our analysis goals and may not suit yours. The full score distribution is available at floor 0.0 if you need higher recall.

Killer whale location data can be sensitive. The MARS deployment location is public and fixed, and the sighting records we cross-reference are already published, so this release adds no new exposure — but a similar pipeline on a mobile or undisclosed deployment might.

---

## Provenance and licensing

Trained on MBARI MARS recordings from the Pacific Ocean Sound archive (public, AWS Open Data). Annotations by D. Edgington (orca, dolphin, ship noise) and J. P. Ryan (humpback), MBARI.

Released under Apache License 2.0. Perch V2 is the work of Google Research under its own license — these weights are a linear probe *on top of* Perch V2 embeddings and do not redistribute any Perch V2 parameters. See `NOTICE`.

Contact: see `CITATION.cff`.
