# labels/

Expert-confirmed annotations for every month used in this work, exported per month and per class.

Definitions and field reference: **[../docs/annotation_schema.md](../docs/annotation_schema.md)** — label vocabulary, taxonomic identifiers, review protocol, and what each field means. The machine-readable structure is `schema.json` (JSON Schema draft 2020-12); validate with:

```bash
pip install jsonschema
python3 ../tools/validate_labels.py --labels .
```

How the release was assembled and what "confirmed" means: [../docs/DATA.md](../docs/DATA.md).

## Quick orientation

| File pattern | Contents |
|---|---|
| `labels_<YYYY>_<MM>_<class>.json` | All annotations of one class in one month |

Each annotation identifies a single 5-second window by recording filename and offset in seconds.

| Month | Annotations | Positive | Weak negative | Role |
|---|---|---|---|---|
| April 2018 | 715 | 661 | 54 | training |
| May 2018 | 283 | 283 | 0 | **held out — never trained on** |
| October 2020 | 322 | 322 | 0 | training |
| April 2026 | 86 | 86 | 0 | training |
| **Total** | **1,406** | **1,352** | **54** | |

## Five things to know before using these

**Every positive label was listened to.** Detections are candidates; labels are expert judgments made on the audio, usually with 30 seconds of context. Where a clip could not be resolved by ear it was left unlabeled rather than forced into a class.

**`humpback_song` means humpback vocalization generally**, not strictly complex song. The name is broader than it sounds. See DATA.md.

**Per-class files mix positives and weak negatives.** `labels_2018_04_orca_call.json` holds 428 entries, but 374 are positives and 54 are weak negatives — background examples used as training signal. Read `n_positive` and `n_negative` from each file rather than its total. April 2018 is the only month with weak negatives.

**April 2018 carried 28 duplicate rows, which are excluded here.** It was the first month analyzed and the month the tooling was built on; repeated `merge_dbs.py` runs re-inserted 14 windows from 30 April up to three times each. Later months are clean. The exporter deduplicates on (recording, offset, label) and reports what it drops.

**Four labels were corrected on 7 September 2026, after the listening page was published.** J. P. Ryan listened to the page, flagged three windows as wrong, and a fourth on re-review; D. Edgington reviewed each in the annotation interface and relabeled:

| Window | Was | Is |
|---|---|---|
| `MARS_20180412_173913` +590 s | `other` | `ship_noise` — vessel propulsion |
| `MARS_20180423_235912` +350 s | `humpback_song` | `dolphin_call` |
| `MARS_20180424_050912` +420 s | `humpback_song` | `dolphin_call` |
| `MARS_20180404_185912` +40 s | *(unlabeled)* | `ship_noise` |

Two of the three errors were humpback labels from April 2018, a month holding only 21 of them against October 2020's 265 — incidental annotations made while working on orca, and correspondingly less scrutinized. The page's sampler has since been changed to draw from the days where a class was most heavily reviewed.

The second reclassification is worth dwelling on rather than hiding: the 24 April window could be orca, and the surrounding context is what settled it as dolphin. Multiple animals in one 5-second window is not an edge case in this archive — it is likely, because the animals interact — and humpbacks can imitate a great deal of what else is in the ocean. Some windows will not resolve to a single species, and this release leaves those unlabeled rather than forcing them.

**April 2026 holds one confirmed orca**, on 24 April, with humpbacks in the background — resolved by ear once comparable examples had accumulated from other years. A second April 2026 candidate remained ambiguous on re-review and is unlabelled rather than forced into a class.
