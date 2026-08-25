# labels/

Expert-confirmed annotations for every month used in this work, exported per month and per class.

Full schema, class definitions, annotator provenance, and what "confirmed" means: [../docs/DATA.md](../docs/DATA.md).

## Quick orientation

| File pattern | Contents |
|---|---|
| `labels_<YYYY>_<MM>_<class>.json` | All annotations of one class in one month |

Each annotation identifies a single 5-second window by recording filename and offset in seconds.

| Month | Annotations | Role |
|---|---|---|
| April 2018 | 685 | training |
| May 2018 | 260 | **held out — never trained on** |
| October 2020 | 317 | training |
| April 2026 | 74 | training |
| **Total** | **1,336** | |

## Three things to know before using these

**Every positive label was listened to.** Detections are candidates; labels are expert judgments made on the audio, usually with 30 seconds of context. Where a clip could not be resolved by ear it was left unlabeled rather than forced into a class.

**`humpback_song` means humpback vocalization generally**, not strictly complex song. The name is broader than it sounds. See DATA.md.

**The April 2026 ambiguous candidates are not in here as confirmed orca.** Thirteen high-scoring April 2026 clips sounded like orca in isolation and showed humpback vocalization throughout their surrounding context; they await a blind second-expert review. Please do not treat them as confirmed.
