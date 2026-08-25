# models/

| File | What |
|---|---|
| `orca_v10.pt` | **Current best.** Use this for new work. |
| `orca_v10.metrics.json` | Per-class F1, thresholds, training arguments, evaluation support |
| `orca_v4.pt` | Prior production model — kept so published figures and the held-out comparison stay reproducible |

Both are linear probes on frozen Perch V2 embeddings — a few hundred kilobytes each. Five output classes: `orca_call`, `humpback_song`, `dolphin_call`, `ship_noise`, `other`.

Read [../docs/MODEL_CARD.md](../docs/MODEL_CARD.md) before using either. The short version:

- Scores are **logits, not probabilities**.
- The inference default floor of 0.0 is far too permissive. Use **+1.16**.
- A single global threshold cannot serve all five classes — per-class optima span roughly +0.2 to +2.5. Take them from `orca_v10.metrics.json`.
- The dominant failure mode is humpback vocalization scoring as orca. Listen to 30 seconds of context, not just the 5-second window.

## Version numbering

The released lineage is v0 → v1 → v2 → v3 → v4 → v10. Versions 5 through 9 are retired experiments, not part of this release; the jump to v10 was deliberate, to remove any ambiguity about which numbers were reused. A separate, earlier `_clean` era of pre-normalization bootstrap models is retired entirely. [../docs/METHOD.md](../docs/METHOD.md) has the full story.
