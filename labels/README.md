# labels/

Expert-confirmed annotations for every month used in this work, exported per month and per class.

Full schema, class definitions, annotator provenance, and what "confirmed" means: [../docs/DATA.md](../docs/DATA.md).

## Quick orientation

| File pattern | Contents |
|---|---|
| `labels_<YYYY>_<MM>_<class>.json` | All annotations of one class in one month |

Each annotation identifies a single 5-second window by recording filename and offset in seconds.

| Month | Annotations | Positive | Weak negative | Role |
|---|---|---|---|---|
| April 2018 | 714 | 660 | 54 | training |
| May 2018 | 283 | 283 | 0 | **held out — never trained on** |
| October 2020 | 322 | 322 | 0 | training |
| April 2026 | 86 | 86 | 0 | training |
| **Total** | **1,405** | **1,351** | **54** | |

## Five things to know before using these

**Every positive label was listened to.** Detections are candidates; labels are expert judgments made on the audio, usually with 30 seconds of context. Where a clip could not be resolved by ear it was left unlabeled rather than forced into a class.

**`humpback_song` means humpback vocalization generally**, not strictly complex song. The name is broader than it sounds. See DATA.md.

**Per-class files mix positives and weak negatives.** `labels_2018_04_orca_call.json` holds 428 entries, but 374 are positives and 54 are weak negatives — background examples used as training signal. Read `n_positive` and `n_negative` from each file rather than its total. April 2018 is the only month with weak negatives.

**April 2018 carried 28 duplicate rows, which are excluded here.** It was the first month analyzed and the month the tooling was built on; repeated `merge_dbs.py` runs re-inserted 14 windows from 30 April up to three times each. Later months are clean. The exporter deduplicates on (recording, offset, label) and reports what it drops.

**April 2026 holds one confirmed orca**, on 24 April, with humpbacks in the background — resolved by ear once comparable examples had accumulated from other years. A second April 2026 candidate remained ambiguous on re-review and is unlabelled rather than forced into a class.
