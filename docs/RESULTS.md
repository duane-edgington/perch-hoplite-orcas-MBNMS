# RESULTS.md — what we found, and how confident we are

All numbers below are reproducible from this repository plus the public raw audio. Where a claim rests on expert listening rather than computation, that is stated. Where something is unresolved, it is listed as unresolved.

---

## The held-out test

**May 2018 is a permanently held-out test month.** No classifier in this release's lineage has ever trained on it, and none ever will. This is a standing policy, not a one-off split: it means any future model can be measured against May with no caveats.

Neither `orca_v4` nor `orca_v10` saw May data during training, which makes it a fair referee for the question that matters — did the labeling loop actually improve the model, or did it just change the evaluation set?

### Recall on 196 ear-confirmed orca windows

Confirmed by listening across 12, 13, 14 and 16 May.

| Metric | orca_v4 | orca_v10 |
|---|---|---|
| Mean logit on confirmed orca | 1.646 | **2.613** |
| Head-to-head wins (195 shared windows) | 3 | **192** |
| Mean difference (v10 − v4) | — | **+0.958** |
| Recall @ +1.16 | 60.7% | **79.6%** |
| Recall @ +2.0 | 39.3% | **59.2%** |
| Confirmed orca scored below 0.0 (missed) | 0 | 1 |
| Total May detections @ floor 0.0 | 241 | 271 |

Reproduce with `tools/compare_may_holdout.py`.

Two honest observations. First, `orca_v10`'s more selective net drops exactly one of 196 known orca that v4's broader net caught — negligible against 192 windows where v10 scores higher, but real. Second, near-equal total detection counts (271 vs 241) matter: v10's higher confidence is not coming from indiscriminate over-firing.

### Precision at the operating threshold

Recall alone can be bought with a looser model, so the complement was measured directly. Of v10's 271 May detections, 195 are already-confirmed orca windows — 195 rather than 196, because v10 scored one confirmed window below the 0.0 floor — leaving **76 unconfirmed**. Of those, **14 scored at or above +1.16**: the precision-critical set. All 14 were reviewed by ear.

**Result: 14/14 real orca. Zero false positives.** No clip was a misfire reclassified to humpback, ship or other.

`orca_v10` therefore wins on both axes — it scores known orca about a full logit higher *and* its high confidence is earned rather than inflated.

### Four new orca days

Among those 14 confirmations were clips on **2, 3, 7 and 29 May 2018** — days with no previously confirmed orca. May's confirmed orca days went from four to eight.

Full May by-day confirmed `orca_call` counts: 2 May = 1, 3 May = 1, 7 May = 1, **12 May = 181**, 13 May = 8, 14 May = 10, 16 May = 7, 29 May = 1.

This is the clearest demonstration of the method's value. The retrained model did not merely rank known calls better — it surfaced real acoustic presence that the previous production model missed entirely, on data neither model had seen.

*Note on held-out status:* these 14 windows were added to the May database **as labels**. Labeling ground truth in a test month is exactly how you measure a model on it. May remains excluded from all training; the policy is unchanged.

---

## Absence, measured

A detector is only as interesting as its silences.

**October 2020.** Bigg's killer whales were visually documented in Monterey Bay, yet the acoustic record shows no confirmed orca vocalizations. Under `orca_v4` this could have been a sensitivity limit. So the more sensitive `orca_v10` — the model demonstrably able to find faint orca that v4 missed on May — was run over all 535,278 windows: 113 detections at floor 0.0, 16 at or above +1.16. All 16 reviewed by ear: **approximately 14 humpback, 2 other, zero orca.**

A promising 4–5 October high-scoring cluster, with consecutive windows in one recording — the profile of a genuine call sequence — turned out to be humpback.

This is the stronger result, not a disappointment. Absence measured with a good instrument is real absence, and it is consistent with Bigg's whales hunting silently: seen but not heard.

**September 2024** offered what should have been an ideal external ground-truth test — a documented all-day, three-matriline Bigg's encounter on 27 September, independently reported. Recording had stopped on 19 September following a power-connector failure. The encounter is not in the acoustic record and is not recoverable from this deployment. Coverage was checked before inference, which is the only reason a meaningless "no orca on 27 September" result was never produced.

Sept 1–19 *was* embedded and scanned. The detection profile — roughly 60 detections at ≥ +2.31 smeared evenly across all covered days at 1–8 per day, with no clustering — is the signature of a persistent weak background source rather than an encounter, and September is peak humpback season. Recorded as **provisional: humpback-heavy, no orca-encounter signature, not confirmed either way** pending an ear review of the top clips.

---

## Presence, confirmed

April 2018 turned out to be a sustained multi-day Bigg's presence rather than the single known event it was assumed to be. Six confirmed orca days: **13, 18, 21, 23, 24, 25 April**, established through systematic ear review of every above-threshold detection.

| Day | Detections @ ≥1.16 (v4) | Review | Outcome |
|---|---|---|---|
| 13 April | 251 | Gradio + J. Ryan | Confirmed Bigg's hunting event (morning) |
| 18 April | 173 | 25/25 reviewed | Confirmed bout (late morning), 0 false positives |
| 21 April | 25 | 25/25 reviewed | Confirmed, no ambiguity |
| 23 April | 39 | 58 across 23–24 reviewed | 55 orca, 2 humpback, 1 mixed left unlabeled |
| 24 April | 19 | (as above) | ~95% orca across both days |
| 25 April | 60 | 50/50 reviewed | Confirmed (evening encounter) |

Independent corroboration is non-acoustic: local news reported record killer whale sightings across April–May 2018, with biologists identifying two pods — one Alaskan, one Californian, the CA140s or "Emma's pod," a Bigg's matriline known for hunting gray whale calves. Two independent methods, hydrophone and visual sighting, agree on a sustained spring 2018 presence.

The CA140 matriline recurs across the archive — spring 2018, October 2020, and September 2024 — which is a large part of why full-archive seasonal and interannual analysis is the natural next step.

Strong `ship_noise` appears in the background throughout the confirmed April orca days: whale-watch and research boats arrive once orcas are spotted. A consequence worth stating plainly for anyone building on this — **ship noise is not an orca-absent cue in this dataset.** Any false-positive suppression logic keying on vessel noise would be exactly backwards on event days.

---

## Classifier metrics

Aggregate metrics across the released lineage:

| Model | ROC-AUC | cmap | Training recipe | Labels | Status |
|---|---|---|---|---|---|
| v0 | 0.9773 | 0.8810 | April 2018 | 584 | baseline |
| v1 | 0.9533 | 0.7999 | + October 2020 | 778 | cross-season |
| v2 | 0.9654 | 0.8930 | April 2018 expanded | ~830 | best for April 2018 |
| v4 | 0.9590 | 0.8297 | 3-season | 803 | **prior production** |
| **v10** | 0.9372* | 0.678* | 3-season, updated labels | 1,076 | **current best** |

\* **Read this before comparing v10's aggregates to v4's.** `orca_v10`'s evaluation set is larger and harder — 459 held-out examples against v4's 296 — so aggregate ROC-AUC and cmap are not directly comparable across the two rows. A bigger, more diverse evaluation set pulls aggregates down even when the model genuinely improves. The held-out May test above is the honest comparison, and per-class F1 is the honest within-model view.

`cmap` is class mean average precision: per-class average precision averaged with equal weight across classes on held-out data (`perch_hoplite.agile.metrics.cmap`, `sample_threshold=1`).

### Per-class F1 for orca_v10, at F1-optimal thresholds

| Class | F1 | Held-out n | Change |
|---|---|---|---|
| `orca_call` | 0.945 | 61 | held steady (~0.95) |
| `ship_noise` | 0.800 | 10 | **first credible score** — previously a fake 1.0 on n=3 |
| `dolphin_call` | 0.687 | 44 | slightly down from ~0.71–0.77 |
| `humpback_song` | 0.619 | 67 | improved from ~0.55 |
| `other` | 0.591 | 22 | current weak point |

`ship_noise` is the clearest labeling win. It sat at n=3 held-out support in every earlier model, and its reported F1 of 1.0 was an acknowledged artifact of that. A targeted campaign took project-wide confirmed ship labels from 35 to 81, producing the first F1 for that class that means anything.

`other` is now the visible weak spot, which is unsurprising: it is a heterogeneous catch-all that grew from 52 to 124 labels while merging three seasons of unclassifiable clips. It plausibly needs its own split, the same way `humpback_song` does.

---

## Known limitations and open questions

**`humpback_song` is a misnomer.** The class covers humpback vocalization generally, not strictly complex song. It lumps true song together with non-song social calls, and that heterogeneity is the leading explanation for why it is the hardest credible class to learn. A gray whale contamination hypothesis was tested directly — two full batches, April 2018 and April 2026, reviewed by the humpback expert — and found **zero** gray whale sounds. That hypothesis is closed, not merely unsupported. Splitting the class into song and non-song is identified but not yet done.

When presenting any humpback clip from this dataset, please distinguish: "song" only for clips plausibly part of a song sequence, "humpback vocalization" or "non-song call" otherwise. The database label name is broader than the acoustic category it is usually assumed to mean.

**Humpback and orca overlap acoustically, and humpbacks can sound like almost anything.** The clearest instance is April 2026: 13 candidates at ≥ +2.31, including the highest-scoring clip anywhere in the project and a near-consecutive bout — on paper the most orca-looking candidate set we have generated. On review, several clips sounded clearly orca within their 5-second window while the surrounding 30 seconds revealed humpback vocalizations throughout. Audio alone cannot separate "orca and humpback both present" from "humpbacks producing orca-like sounds within their own repertoire."

**These April 2026 candidates are recorded as ambiguous and are NOT included as confirmed orca anywhere in this release.** A blind second-expert review is the required next step; agreement would make them strong, divergence would confirm genuine ambiguity. Please do not cite them as confirmed orca days.

**A single global threshold does not work.** Per-class F1-optimal thresholds span roughly +0.2 to +2.5. The inference default of 0.0 is uniformly too permissive.

**Precision is measured at and above threshold on confirmed sets, not as a month-wide false-alarm rate.** The May check is clean and complete for what it covers; it is not an exhaustive negative labeling of the month.

**Ecotype classification is not provided.** The orca detected here are predominantly Bigg's (transients), largely CA140-associated. The models are not validated on Residents or Offshores, and no ecotype discrimination is offered. This is future work.

**Everything here is MARS and Monterey Bay.** Single hydrophone, 891 m depth, one bay. Transfer to other deployments is plausible but untested — and testing it is exactly the kind of thing we hope someone does with this release.

**Perch V2 embeds species and collapses within-orca variation.** An intriguing by-day t-SNE result — the 25 April evening encounter separating cleanly from the 13 April morning event within the same month, robust across perplexity 10/30/50 and across 10 recordings spanning 3.5 hours — is a lead, not proof of pod or individual structure. Whether the separating clusters correspond to the two known pods is a testable question we have not answered.

---

## Label totals

| Database | Annotations |
|---|---|
| April 2018 | 685 |
| May 2018 | 260 |
| October 2020 | 317 |
| April 2026 | 74 |
| **Total** | **1,336** |

Two counts appear in our materials and both are correct because they answer different questions: **873** labels built the trajectory of the presented v4 model; **1,336** is everything confirmed across all months and classes to date, including work completed after v4 was trained. They should be stated separately rather than reconciled by rounding.

Total expert annotation effort to `orca_v10`: roughly 8–10 hours of listening across several weeks. That is the number worth dwelling on. Labeling was always the bottleneck — never computation.
