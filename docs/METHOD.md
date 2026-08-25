# METHOD.md — how the classifier was built

The agile modeling loop, as it actually ran. This is the narrative version; the numbers it refers to are in [RESULTS.md](RESULTS.md) and the operating details are in [MODEL_CARD.md](MODEL_CARD.md).

---

## The loop

```
Embed audio  →  Search DB  →  Human review (Gradio)  →  Label
     ↑                                                     ↓
     └────  Review detections  ←  Infer  ←  Train classifier
```

The whole approach rests on one asymmetry: **embedding is expensive and happens once; everything after it is nearly free.** A month of audio takes about 40 minutes to embed on a GB10. Training a linear probe on top of those frozen embeddings takes about 30 seconds. So a labeling session in the morning can produce a measurably better model before lunch, and the bottleneck becomes what it should be — expert listening.

Sessions ran 25 clips at a time (initially 50, reduced because 25 is about the attention span for careful listening), with larger batches stepped through in 25-clip chunks. Each clip is a 5-second window, presented with a spectrogram, playable audio, and 30 seconds of surrounding context.

Labeling split by expertise: D. Edgington annotated orca, ship noise and dolphin; J. Ryan, the humpback expert, annotated humpback. That division matters for provenance — see [DATA.md](DATA.md).

---

## Where the candidates came from

This is the single most important thing to understand about the method, and it changed early.

The Google Multispecies Whale Model's score CSVs were used for **one thing only**: an initial 100-clip screen — 50 high-scoring orca candidates and 50 low-scoring background candidates. Those 100 hand-screened annotations trained the first real classifier.

**From v0 onward, every candidate came from the Perch V2 classifier's own detections.** Whenever this documentation says "review of top-scoring detections," it means top-scoring under the Perch-based classifier being iterated — run over the audio, ranked by its own scores, with the highest-scoring and the confidently-wrong both pulled for review. Once a Perch-based classifier existed, the project fed exclusively on its own output. That self-feeding loop is the method.

---

## The fix that made it work

MARS recordings at 891 m have typical peak amplitudes of 0.001–0.003 — very quiet. Without per-window peak normalization to 0.25 before the model, the PyTorch Perch V2 port diverged from the TensorFlow reference at **cosine 0.43–0.94** on real MARS audio. Not a subtle drift; effectively different embeddings.

The fix is one line conceptually — normalize each 5-second window to peak 0.25 before embedding — and it unlocked everything downstream. All embeddings were regenerated afterward. Databases built this way carry a `_norm` suffix by convention.

Anyone applying Perch V2 (or a similar frozen embedding model) to quiet hydrophone audio should check this first. It is the kind of bug that produces plausible-looking, entirely wrong results.

---

## Version lineage

The released lineage is **v0 → v1 → v2 → v3 → v4 → v10**. Five classes throughout: `orca_call`, `humpback_song`, `dolphin_call`, `ship_noise`, `other`, plus a weak-negative label for background.

### v0 — baseline
April 2018 only, 584 labels. Established the five-class structure and identified the 13 April Bigg's hunting event. ROC-AUC 0.9773. Twenty-two seconds to train.

### v1 — add a season
Plus October 2020: 209 humpback and 5 dolphin labels from peak humpback season. ROC-AUC fell to 0.9533 — cross-season generalization is genuinely harder, and the drop is informative rather than a failure.

### v2 — add balance
More dolphin and `other` examples in April 2018. Best model for April/May 2018 specifically. ROC-AUC 0.9654.

### v3 and v4 — hard negative mining
April 2026 was added, and this is where the loop paid off most visibly. The top-25 highest-scoring *orca* detections in April 2026 were reviewed and **every one was humpback song misclassified as orca.** Relabeled as hard negatives rather than discarded, they cut April 2026 orca false positives from 6,489 to 304 — a 95% reduction — from 17 labels.

A second round added 8 more hard negatives, giving **v4**: ROC-AUC 0.9590, cmap 0.8297, 803 labels, best cross-season classifier and production model through August 2026.

Confidently-wrong predictions are the highest-value labels available. Finding them costs one review session.

### v5 through v9 — retired
Not in this release. Briefly, for honesty and so nobody repeats them:

A context-embedding experiment replaced raw 5-second embeddings with 30-second Gaussian-weighted averages. The t-SNE looked beautiful — orca and humpback fully separated — and the classifier got materially worse (cmap 0.830 → 0.595). A context post-processing filter suppressed genuine 13 April orca. The reason is biological: **Bigg's killer whale calls are brief discrete bursts, not sustained bouts.** Averaging over 30 seconds dilutes exactly the signal you want. A pretty embedding visualization is not a better classifier.

Adding May 2018 as a fourth training season inflated `ship_noise` detections roughly fourfold across three attempted fixes, none of which worked. The leading explanation is that the two spring events are acoustically distinct enough to spread the orca embedding cluster and shift neighbouring decision boundaries.

Those experiments informed what came next; their weights are not part of the released lineage.

### The diagnostic turn

After v6–v8 exhausted "add more data," the work shifted from training to **measuring**. Per-class F1 on the same held-out split as cmap replaced aggregate metrics as the primary read, and the picture changed immediately: `orca_call` was strong but needed a positive threshold; `humpback_song`, once it had real held-out support, was the weakest credible class at ~0.55; and `ship_noise`'s perfect 1.0 was exposed as an artifact of n=3.

Cross-month validation over four ground-truth months established the operating threshold. False positives collapse under thresholding — October 2020 from 144 to 1, April 2026 from 323 to 6, across logit 0.0 to +2.0 — while confirmed events retain most of their detections. **+1.16 became the operating threshold**; the default 0.0 is unusable.

Aggregate metrics tell you a model changed. Per-class metrics tell you what to do next.

### Closing the measured gaps → v10

The measured weaknesses became a punch list rather than a caveat section:

- The last unreviewed April orca day (21 April) was resolved — 25/25 confirmed orca.
- The gray whale contamination hypothesis for humpback's weak F1 was tested directly across two full batches and **closed with zero contamination found**. The real cause was reframed: `humpback_song` lumps song with non-song vocalizations.
- `ship_noise` got a targeted labeling campaign: 35 → 81 confirmed labels project-wide.
- May 2018 was set aside as a permanent held-out test month rather than folded back in — learning from the four-season lesson instead of repeating it.

The retrain used **v4's exact three-season recipe** (April 2018 + October 2020 + April 2026) on the fully updated label set: 1,076 annotations against v4's 803. Same recipe, better labels — a clean before-and-after with no confounding change in season mix.

That model is `orca_v10`. Its aggregate ROC-AUC reads lower than v4's, because the evaluation set grew from 296 to 459 examples and got harder alongside the training set. Per-class F1 and the held-out May test both show real gains. See [RESULTS.md](RESULTS.md).

*On the numbering:* v5 through v9 are all considered taken or retired, so the next model deliberately started at v10 to remove ambiguity. The pre-normalization bootstrap models (`v1_clean`–`v8_clean`, auto-seeded from Google model scores with no human review) are a separate retired era entirely, and there was never a `v0_clean` — v0 was already the first post-normalization, human-labeled model.

---

## The arc, in one paragraph

Bootstrap from an existing model's scores → fix the normalization bug that made everything downstream wrong → expand to five classes → expand by season, by balance, and by hard negatives → hit a ceiling and *diagnose it* with per-class metrics instead of adding more data → close the specific gaps the diagnosis identified → retrain and validate on a month never trained on.

One expert, roughly 8–10 hours of listening spread over weeks, produced a cross-season Bigg's killer whale detector — and an instrument that surfaced testable marine mammal biology, including a multi-day April presence and per-encounter acoustic structure, with its limits measured rather than hidden.
