# Annotation Schema — MARS Hydrophone Bioacoustic Labels

**Repository:** <https://github.com/duane-edgington/perch-hoplite-orcas-MBNMS>
**Zenodo (concept DOI, always the newest version):** <https://doi.org/10.5281/zenodo.22574816>
**Contact:** D. Edgington, MBARI (duane@mbari.org)

This document defines the label vocabulary and the meaning of each field. The
machine-readable structure of the label files is `labels/schema.json` (JSON Schema
draft 2020-12), which `tools/validate_labels.py` checks them against. What the
release contains and how it was assembled is in [DATA.md](DATA.md); model behavior
and thresholds are in [MODEL_CARD.md](MODEL_CARD.md).

---

## 1. Recording and Windowing

All annotations apply to 5-second non-overlapping windows extracted from continuous
hydrophone recordings at the MARS cabled observatory node.

| Parameter | Value |
|---|---|
| Hydrophone location | 36° 42.7481′ N, 122° 11.2139′ W (36.71247°, −122.18690°), 891 m depth |
| Raw recording format | 256 kHz, 24-bit, mono, 10-minute files |
| Sample rate (resampled) | 32,000 Hz, 16-bit |
| Amplitude handling | SoX `vol 3` (voltage calibration, see below); per-window peak normalization to 0.25 applied before embedding |
| Window duration | 5 seconds |
| Window stride | 5 seconds (non-overlapping) |
| Embedding model | Perch V2 (Google Research), PyTorch port by D. Edgington, MBARI |
| Embedding dimension | 1,536 |
| Classifiers | `orca_v4.pt`, `orca_v10.pt` — linear probes on frozen Perch V2 embeddings |

**Two distinct amplitude operations, often conflated.** `vol 3` in the resampling
step is a **calibration to volts**, the physical unit the science works in, and is
applied to every 32 kHz resample in this project by design. It is not a loudness
adjustment for the model's benefit. Separately, MARS recordings at 891 m are very
quiet — typical peak amplitudes of 0.0015 to 0.003 — and that is handled by
**per-window peak normalization to 0.25** immediately before embedding. Without that
normalization the PyTorch Perch V2 port diverges from the TensorFlow reference at
cosine 0.43–0.94 on real MARS audio. See [../resampling/README.md](../resampling/README.md).

---

## 2. Label Vocabulary

Five classes appear in this release. Each label applies to one 5-second window.

### 2.1 Positive labels (`label_type = "positive"`, `label = 1`)

#### `orca_call`

**Taxon:** *Orcinus orca* (Linnaeus, 1758) — Bigg's / transient ecotype
**WoRMS AphiaID:** 137102 · **ITIS TSN:** 180469 · **GBIF taxon key:** 2433718

**Definition:** A window containing one or more vocalizations attributable to killer
whales, confirmed by expert human review via spectrogram inspection and audio
playback. Killer whales produce tonal calls, pulsed calls, and clicks. The
diagnostic feature used in this dataset is a tonal call with characteristic harmonic
structure, often with a frequency sweep or downturn at the end, distinct from
dolphin calls.

**Ecotype note:** Detections in this dataset are attributed to the Bigg's
(transient, mammal-hunting) ecotype on the basis of geographic context (Monterey
Bay, California), prey availability (gray whale calves, dolphins, pinnipeds), and
behavioral context from concurrent surface sightings. Resident (fish-eating) killer
whales are rare in Monterey Bay. **The attribution is contextual, not acoustic: the
released classifiers perform no ecotype discrimination and are not validated on
Residents or Offshores.**

**Labeling criteria:**
- Call structure visible in the mel spectrogram (10 Hz floor, viridis colormap)
- Confirmed by ear in both the 5-second clip and the 30-second context window
- The reviewer exercises judgment where the soundscape is complex, which in this
  archive is common rather than exceptional — humpback, dolphin and killer whale
  can co-occur within one window because the animals interact
- Windows that cannot be resolved to a single species are **left unlabeled** rather
  than assigned to `orca_call` or to `other`

---

#### `humpback_song`

**Taxon:** *Megaptera novaeangliae* (Borowski, 1781)
**WoRMS AphiaID:** 137119 · **ITIS TSN:** 180530 · **GBIF taxon key:** 2440484

**Definition:** A window containing humpback whale vocalization, including song
units, social calls, or feeding calls, spanning roughly 20 Hz to 8 kHz. In Monterey
Bay humpbacks are acoustically active year-round with peak activity in summer and
fall.

**The class name is broader than it sounds.** It covers humpback vocalization
generally, not strictly song, and mixes true song sequences with non-song social
calls. That heterogeneity is the leading explanation for this being the hardest
credible class to learn. When describing an individual clip, say "humpback
vocalization" or "non-song call" unless it is plausibly part of a song sequence — do
not caption a social call as "song" merely because the database label reads
`humpback_song`.

---

#### `dolphin_call`

**Taxon:** Delphinidae (family level; species not resolved)
**WoRMS AphiaID:** 136980 (family Delphinidae)

**Definition:** A window containing delphinid vocalization — clicks, whistles, or
burst-pulse sounds — not attributable to killer whale. Species common in Monterey
Bay include Pacific white-sided dolphin (*Lagenorhynchus obliquidens*) and Risso's
dolphin (*Grampus griseus*). **Species-level identification is not attempted, and no
species-level identifiers are asserted.**

**Note:** Dolphin and killer whale calls can occur in the same 5-second window.
Where both are present and separable, the window is labeled by the dominant or most
clearly identifiable signal; where they are not separable, the window is left
unlabeled.

---

#### `ship_noise`

**Definition:** A window dominated by anthropogenic vessel noise — propulsion,
cavitation, or broadband mechanical noise from passing ships or boats.
Characterized by elevated low-frequency energy below about 500 Hz with broadband
extension. Not a taxon.

**Note for anyone building on this:** in this dataset `ship_noise` sometimes
correlates positively with killer whale presence, because whale-watch and research
vessels arrive once orcas are sighted. It is not an orca-absent cue, and
false-positive suppression keyed on vessel noise would be backwards on event days.

---

#### `other`

**Definition:** A window containing a detectable acoustic signal — biological or
anthropogenic — that fits none of the classes above. Used for signals the reviewer
can confirm are present but cannot assign to a named class.

**Not the same as silence.** Quiet background is a weak negative (below) and never
appears as a positive label. `other` is a heterogeneous catch-all by construction
and is the weakest-performing class in the released models; treat an `other`
detection as "something is here, unclassified" rather than as a meaningful category.

**What `other` typically holds here.** Much of it is biological: a vocalization the
annotator could hear clearly and was confident came from an animal, but could not
identify. Identification beyond the named classes is outside the annotators'
expertise, and no guess is recorded. An `other` label asserts two things and nothing
further — a real signal is present, and it is not one of the named classes.

Windows where the reviewer is uncertain whether a signal is present at all, or
cannot resolve competing interpretations, are left unlabeled rather than placed in
`other`.

---

### 2.2 Weak negative labels (`label_type = "negative"`, `label = 0`)

#### `orca_call` (negative)

**Definition:** A window explicitly confirmed **not** to contain a killer whale
call, used as a hard negative in classifier training. These are windows that scored
above a preliminary detection threshold and were rejected on expert review.

They appear in the database with `label_type = 2` and in the exported files as
`label_type: "negative"`, and are excluded from the positive annotation counts
reported in the poster and papers. This release contains **54** of them, all in
April 2018 `orca_call`, from the original training sessions.

---

### 2.3 Not stored: unlabeled windows

A window the reviewer chose not to label — too faint, too ambiguous, or obscured —
produces **no row** in the database and no entry in the exported files. "Unlabeled"
is therefore the absence of an annotation, not a label value, and unlabeled windows
are excluded from precision and recall calculations.

This is a deliberate discipline rather than an oversight. Leaving a hard window
unlabeled means a future reviewer, better calibrated by more examples, meets it
clean. One April 2026 candidate was resolved as killer whale that way, months after
first review, once comparable orca-over-humpback examples had accumulated.

---

### 2.4 A class defined but not present in this release

#### `ROV_noise`

Noise from a remotely operated vehicle operating at or near the MARS node; MBARI
deploys ROVs periodically for maintenance. Characterized by high-frequency motor
whine and thruster noise, distinct from surface vessel signatures.

**This class has zero annotations in the four released months** and is not an output
class of either released model. It is documented here because it is available in the
annotation interface and appears in later review sessions outside this release.

---

## 3. Annotation Provenance

**D. Edgington (MBARI Senior Software Engineer) annotated every label in this
release. J. P. Ryan verified every humpback label**, and was consulted on ambiguous
clips of other classes.

The exported `annotator` field therefore reads `duane` throughout. It is set by the
exporter rather than read from the database, for reasons specific to these four
months: early review sessions ran with a generic annotator id (`analyst`) rather
than a personal one, and repeated runs of the annotation-merge tool then appended
`_merged` to that value, so some April 2018 rows read
`gradio_gui:analyst_merged_merged_merged`. Newer databases record per-annotator
identity correctly — the review tool writes whatever `--annotator-id` supplies, and
2016 sessions show `gradio_gui:duane` — but the released months predate that
practice. The statement above is the accurate provenance record; the database column
is not.

| Provenance string in the database | Meaning |
|---|---|
| `gradio_gui:analyst` | D. Edgington, using a generic session id |
| `gradio_gui:analyst_merged`, `..._merged_merged`, `..._merged_merged_merged` | The same, rewritten by repeated merges |
| `gradio_gui:duane` | D. Edgington, later sessions (not in this release) |

---

## 4. Review Protocol

1. **Inference.** The classifier scores every 5-second window in the month.
2. **Pass 1.** Windows at or above the model's operating threshold are presented for
   review in descending score order via the Gradio labeling interface.
3. **Pass 2, selectively.** Where pass 1 confirms an encounter but yields few calls,
   review extends to lower scores — sometimes below logit 0.0 — to recover
   additional calls within the known encounter window. The threshold defines a
   review queue, not a detection boundary.
4. **Context.** Each candidate is presented as a 5-second mel spectrogram with audio
   playback, plus a 30-second context window. The context is used to judge whether a
   call is isolated or embedded in a multi-species soundscape, whether it belongs to
   one animal's bout, and whether vessel noise is masking it. It routinely changes
   the reading.
5. **Save.** Labels are written to the database on each click.

**Thresholds are per model and are not interchangeable.** The F1-optimal `orca_call`
threshold is **+1.16** for `orca_v4` and **+2.31** for `orca_v10`; at the latter,
`orca_v10` achieves precision 0.909 and recall 0.984 for `orca_call` on its
459-example held-out evaluation split. Per-class thresholds for both models, and the
reasons a single global threshold cannot serve all five classes, are in
[MODEL_CARD.md](MODEL_CARD.md). Take the numbers from
`models/orca_v10.metrics.json` and `models/orca_v4.metrics.json` rather than from
prose.

Independent validation uses **May 2018**, a permanently held-out month never trained
on by any model in the released lineage. See [RESULTS.md](RESULTS.md).

---

## 5. Field Reference

Field-by-field types, patterns and constraints are in `labels/schema.json`. In brief,
each entry in a label file's `annotations` array carries:

| Field | Meaning |
|---|---|
| `species` | The label class (section 2) |
| `recording_32khz` | Bare filename of the resampled recording; no path |
| `annotation_offset_s` | Window start in seconds from the recording start; a multiple of 5 |
| `frame_index` | `annotation_offset_s / 5` |
| `recording_start_utc_epoch` | Unix epoch of the recording start, from the filename timestamp; add the offset for the window's own UTC time |
| `label` | 1 positive, 0 weak negative |
| `label_type` | `"positive"` or `"negative"` |
| `annotator` | `duane` throughout this release (section 3) |
| `month` | `YYYY_MM`, matching the file |

Validate the files against the schema with:

```bash
pip install jsonschema
python3 tools/validate_labels.py
```

That also cross-checks each file's summary counts against its own contents, and exits
non-zero on any failure.

---

## 6. Taxonomy Reference URIs

| Label | Taxon | WoRMS | ITIS | GBIF |
|---|---|---|---|---|
| `orca_call` | *Orcinus orca* | [137102](https://www.marinespecies.org/aphia.php?p=taxdetails&id=137102) | [180469](https://www.itis.gov/servlet/SingleRpt/SingleRpt?search_topic=TSN&search_value=180469) | [2433718](https://www.gbif.org/species/2433718) |
| `humpback_song` | *Megaptera novaeangliae* | [137119](https://www.marinespecies.org/aphia.php?p=taxdetails&id=137119) | [180530](https://www.itis.gov/servlet/SingleRpt/SingleRpt?search_topic=TSN&search_value=180530) | [2440484](https://www.gbif.org/species/2440484) |
| `dolphin_call` | Delphinidae (family) | [136980](https://www.marinespecies.org/aphia.php?p=taxdetails&id=136980) | — | — |

`ship_noise` and `other` are not taxa and have no taxonomic identifier.

---

## 7. Citation

Cite the archive by its concept DOI, which always resolves to the newest version:

> Edgington, D. R., & Ryan, J. P. (2026). *perch-hoplite-orcas-MBNMS:
> expert-confirmed Bigg's killer whale annotations, trained classifiers, and a
> reproducibility bundle for the MBARI MARS hydrophone archive* [Data set]. Zenodo.
> <https://doi.org/10.5281/zenodo.22574816>

To pin an exact release, use the version DOI shown in the Versions panel of that
record.

And the associated poster:

> Edgington, D. R., & Ryan, J. P. (2026). Application of Foundation Model and Agile
> Modeling to Passive Acoustic Detection of Orcas in Monterey Bay National Marine
> Sanctuary. *IEEE OCEANS 2026*, Monterey.

The underlying raw audio should be cited independently as the Pacific Ocean Sound /
MBARI MARS dataset on the AWS Open Data registry; see
[../data_access/how_to_get_raw_audio.md](../data_access/how_to_get_raw_audio.md).
