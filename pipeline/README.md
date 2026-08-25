# pipeline/

The two programs that do the work.

## `phase1_embed_torch.py` — audio to embeddings

Walks a directory of resampled 32 kHz WAV files, cuts them into 5-second windows, embeds each with Perch V2, and writes a hoplite vector database.

Requires the **perch-pytorch** environment, not this repository's — see [../docs/REPRODUCE.md](../docs/REPRODUCE.md) stage 3.

```bash
python3 phase1_embed_torch.py \
    --audio-dir /path/to/resampled_32kHz/2018/05 \
    --db-dir /path/to/db/MARS_20180501_20180531_32kHz_norm \
    --device cuda --compile
```

Per-window peak normalization to 0.25 is applied inside the embedding adapter. It is not a flag and cannot be disabled, because without it Perch V2 embeddings of quiet MARS audio are wrong. The `_norm` suffix on the database directory is a manual naming convention marking databases built this way — please keep using it.

Re-running is idempotent; already-embedded files are skipped.

## `phase2_classify.py` — everything else

One CLI, several subcommands, covering the whole agile modeling loop after embedding.

| Subcommand | Does |
|---|---|
| `search` | Embed a query clip, find nearest neighbours in the database |
| `label` | Import a CSV of labels into the database |
| `train` | Train a linear probe on the current labels |
| `review` | Serve the Gradio labeling and review interface |
| `infer` | Score every embedding, write a detections CSV |
| `stats` | Print database statistics |

```bash
python3 phase2_classify.py infer \
    --db-dir /path/to/db/MARS_20180501_20180531_32kHz_norm \
    --classifier ../models/orca_v10.pt \
    --labels orca_call --logit-threshold 1.16 \
    --output-csv may2018_v10_orca.csv
```

Run any subcommand with `--help` for its full flag list. Usage examples are also in the module docstring at the top of the file.

`src/` holds the supporting modules: spectrogram rendering, audio encoding and 30-second context, model loading, training, inference, the Gradio review interface, and per-class F1 metrics.

## Notes

Scores are logits, not probabilities. The default floor of 0.0 is far too permissive — see [../docs/MODEL_CARD.md](../docs/MODEL_CARD.md) for thresholds.

Inference output is per-label: one row per window per class. Raw row counts at floor 0.0 are inflated by also-ran classes and should not be read as detection counts.

For the review interface, use **Chrome in incognito mode**. A normal window can serve stale cached state, making the page appear to hang while the server is perfectly healthy. Safari fails separately on audio playback with data URIs.

No TensorFlow is needed anywhere. A TF mock is injected to satisfy upstream perch-hoplite imports.

Path constants in docstrings reflect the MBARI environment. Nothing here requires that layout — all paths are command-line arguments.
