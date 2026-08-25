# tools/

Analysis and evaluation scripts. Each runs standalone; several have paths defined as constants at the top that you will want to edit for your own layout.

## Evaluation

**`compare_may_holdout.py`** — the headline experiment. Scores `orca_v4` and `orca_v10` against the ear-confirmed orca windows in the held-out May 2018 month and reports recall at 0.0 / +1.16 / +1.5 / +2.0 plus score distributions. Reproduces the comparison table in [../docs/RESULTS.md](../docs/RESULTS.md). Needs both models' floor-0.0 inference CSVs and the May database.

A window absent from a model's CSV means that model scored it below 0.0 — a miss at any positive threshold.

**`score_orca_regions.py`** — threshold sweep against known ground-truth regions. This is how the +1.16 operating threshold was established: false positives collapse under thresholding on confirmed-silent months while confirmed events retain most of their detections.

## Labels

**`export_labels.py`** — flatten the SQLite `annotations` tables to per-month, per-class JSON. Schema in [../docs/DATA.md](../docs/DATA.md). Database paths are constants at the top.

**`extract_example_clips.py`** — pull the audio for specific labeled windows into standalone WAV files. Used to build the confirmed-clip subset in the Zenodo record.

**`merge_dbs.py`** — combine embedding databases, offsetting window IDs so they don't collide. This is how the multi-season training databases were built. Merging roughly 1.5 M vectors takes about 20 minutes, mostly index rebuild.

## Plots

**`plot_monthly.py`** — per-day detection counts and heatmap for a month's inference CSV. Deduplicates on `(idx, label)`.

**`plot_tsne.py`** — t-SNE of embeddings, optionally across several databases.

**`plot_tsne_orca_by_day.py`** — confirmed orca embeddings coloured by day. This produced the result that the 25 April evening encounter separates from the 13 April morning event within the same month. Takes `--confirmed-april-days` / `--confirmed-may-days` on the command line, and `--dpi` for print-quality export.

A caution on that last one, learned the hard way: a t-SNE separation is a lead, not evidence, until it clears three checks — same-month comparison (rules out season and background effects), spread across distinct recordings (rules out a single-recording or single-boat artifact), and robustness across perplexity 10/30/50 (rules out a t-SNE artifact). Perch V2 embeds species and collapses within-orca variation, so what these plots can support is bounded.
