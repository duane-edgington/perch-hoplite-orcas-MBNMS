# models/

| File | What |
|---|---|
| `orca_v10.pt` | **Current best.** Use this for new work. |
| `orca_v10.metrics.json` | Per-class F1, thresholds, training arguments, evaluation support |
| `orca_v4.pt` | Prior production model — kept so published figures and the held-out comparison stay reproducible |

Both are linear probes on frozen Perch V2 embeddings — a few hundred kilobytes each. Five output classes: `orca_call`, `humpback_song`, `dolphin_call`, `ship_noise`, `other`.

Read [../docs/MODEL_CARD.md](../docs/MODEL_CARD.md) before using either. The short version:

- Scores are **logits, not probabilities**.
- The inference default floor of 0.0 is far too permissive.
- **Each model has its own orca threshold: +1.16 for `orca_v4`, +2.31 for `orca_v10`.** Not interchangeable.
- Within a model, one threshold cannot serve all five classes — per-class optima span +0.79 to +2.31. Take them from the model's own `.metrics.json`.
- Running both models on a month and comparing is the project's standard practice. See [../docs/MODEL_CARD.md](../docs/MODEL_CARD.md).
- The dominant failure mode is humpback vocalization scoring as orca. Listen to 30 seconds of context, not just the 5-second window.

## Version numbering

The released lineage is v0 → v1 → v2 → v3 → v4 → v10. Versions 5 through 9 are retired experiments, not part of this release; the jump to v10 was deliberate, to remove any ambiguity about which numbers were reused. A separate, earlier `_clean` era of pre-normalization bootstrap models is retired entirely. [../docs/METHOD.md](../docs/METHOD.md) has the full story.
