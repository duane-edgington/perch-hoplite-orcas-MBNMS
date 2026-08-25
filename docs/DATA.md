# DATA.md — what is released, and in what form

## The principle: publish reproducibility, not a dataset

The raw MARS audio is already public, free, and hosted on AWS Open Data. Re-hosting terabytes of it would add cost and no value. The intermediate products are large and fully regenerable: roughly 723 GB of resampled WAV and 45 GB of embedding databases, all derivable from a shell script and a manifest.

So this release publishes the small, irreplaceable things — under 10 MB in total — plus everything needed to regenerate the large things exactly:

| Artifact | Size | Where |
|---|---|---|
| Trained classifiers (`orca_v4.pt`, `orca_v10.pt`) + metrics | ~700 KB | this repo, `models/` |
| Confirmed annotations, all months and classes | ~700 KB | this repo, `labels/` |
| Confirmed example clips (WAV) | ~3 MB | Zenodo |
| Resampling script, source manifest, checksums, pinned versions | KB | this repo |
| Full resampled audio | ~723 GB | **not published** — regenerable |
| Embedding databases | ~45 GB | **not published** — regenerable |
| Raw MARS audio | TB | **already public** — AWS Open Data |

A re-runner goes: public raw audio → our resampling script at our pinned SoX version → verify against our checksums → embed → our models → our results. That is genuine reproducibility, and it costs nobody a hosting bill.

---

## FAIR mapping

**Findable.** Zenodo record with a DOI, linked from this repository and from the poster QR code. Descriptive metadata and keywords: Perch V2, passive acoustic monitoring, killer whale, *Orcinus orca*, Bigg's, transient, agile modeling, MBARI MARS, Monterey Bay National Marine Sanctuary.

**Accessible.** Everything retrievable with standard tools over open protocols — `git clone` for code, Zenodo HTTPS for the archived snapshot, documented `aws s3` or HTTPS paths for the raw audio. No credentials, no gatekeeping, no request forms.

**Interoperable.** Open formats throughout: CSV and JSON for labels and metadata, WAV for clips, PyTorch `.pt` for models, plain Python for code, Raven-compatible selection tables where relevant. No proprietary containers.

**Reusable.** One clear license (Apache 2.0), provenance recorded for every artifact, exact tool versions and flags pinned, checksums so a re-runner can verify they reproduced our inputs rather than silently diverging, and a model card documenting intended use and limits.

---

## Label schema

Labels live in a SQLite `annotations` table inside each embedding database. The relevant shape:

```sql
CREATE TABLE recordings (
    id INTEGER PRIMARY KEY,
    filename TEXT NOT NULL,
    deployment_id INTEGER, ...);

CREATE TABLE annotations (
    id INTEGER PRIMARY KEY,
    recording_id INTEGER NOT NULL REFERENCES recordings(id),
    offsets FLOAT_LIST NOT NULL,     -- window start, seconds into the recording
    label TEXT NOT NULL,             -- orca_call | humpback_song | dolphin_call | ship_noise | other
    label_type INTEGER NOT NULL,     -- 1 = positive, 2 = weak negative
    provenance TEXT NOT NULL);       -- annotator / session identifier
```

`tools/export_labels.py` flattens this to per-month, per-class JSON:

```json
{
  "month": "2018_05",
  "species": "orca_call",
  "count": 210,
  "n_positive": 210,
  "n_negative": 0,
  "n_recordings": 42,
  "annotations": [
    {
      "species": "orca_call",
      "recording_32khz": "MARS_20180512_083912_resampled_32kHz.wav",
      "annotation_offset_s": 195.0,
      "frame_index": 39,
      "recording_start_utc_epoch": 1526114352,
      "label": 1,
      "label_type": "positive",
      "annotator": "duane",
      "month": "2018_05"
    }
  ]
}
```

Each annotation identifies a single 5-second window by recording filename and offset. `frame_index` is `offset / 5`. Times derive from the MARS filename convention `MARS_<YYYYMMDD>_<HHMMSS>` and are UTC.

---

## Label classes

| Label | Type | Meaning |
|---|---|---|
| `orca_call` | positive | Killer whale vocalization. In this dataset, predominantly Bigg's (transient). |
| `humpback_song` | positive | **Humpback vocalization generally**, not strictly complex song — see the caution below. |
| `dolphin_call` | positive | Delphinid vocalization other than killer whale. |
| `ship_noise` | positive | Vessel noise. Correlates *positively* with orca presence here — boats follow sightings. |
| `other` | positive | Real acoustic content that fits no named class: mixed signals, unusual sounds, ambiguous calls. Heterogeneous by construction. |
| `negative` | weak negative | Background and ambient ocean noise — nothing biological happening. Training signal only; does not appear as a detection class in inference output. |

The key distinction people get wrong: **`negative` means silence or background** and is a weak negative used only in training. **`other` means a real sound that couldn't be classified** and is a positive class that does appear in inference output.

**Caution on `humpback_song`.** The name is broader than it sounds. It covers humpback vocalization generally, including non-song social calls, because a 5-second focal window counts as humpback if it is part of a humpback sequence. When you report on a clip, say "humpback vocalization" or "non-song call" unless the clip is plausibly part of an actual song sequence. Do not caption a non-song clip as "song" merely because the database label reads `humpback_song`.

---

## What "confirmed" means

Every positive label in this release was **listened to by an expert annotator**. Model detections are candidates; labels are judgments made by ear on the audio, usually with 30 seconds of surrounding context.

Annotator provenance, which matters for interpreting the labels:

- **D. Edgington** — orca, ship noise, dolphin. Killer whale identifications are his expert call.
- **J. P. Ryan** — humpback. He is the humpback expert on the project; humpback identifications and quality-assurance reviews are his.

Where a clip could not be resolved by ear, **it was left unlabeled rather than forced into a class.** This happened deliberately and repeatedly. One October 2020 review session produced 25 candidates and 0 labels, because every one showed audible contamination and none could be honestly called. That is a feature of the labeling discipline, not a gap in it.

**Not included as confirmed:** the April 2026 candidates. Thirteen high-scoring clips there sounded like orca in isolation, with humpback vocalization throughout their surrounding context. They await a blind second-expert review and are documented in [RESULTS.md](RESULTS.md) as ambiguous. Please do not treat them as confirmed orca.

---

## Zenodo archive

A versioned, DOI'd snapshot of the derived artifacts — trained models, label tables, the confirmed-clip audio subset, and a frozen copy of this repository at release — is archived on Zenodo. That record is what a paper should cite.

Zenodo is free and its standard per-record quota is comfortably larger than this release needs; the entire payload is single-digit megabytes.

**DOI:** *to be minted at release.*

Only derived artifacts go to Zenodo. No raw audio, no resampled months, no embedding databases.

---

## Licensing

**Apache License 2.0** covers code, models, and released label data alike. See `LICENSE`.

Two things are not ours to relicense:

- **Raw MARS audio** is distributed via AWS Open Data under its own terms. Link and cite it; do not re-host or relicense it.
- **Perch V2** is Google Research's model under its own license. The classifiers released here are linear probes trained *on top of* Perch V2 embeddings and contain no Perch V2 parameters, but the dependency should be acknowledged — see `NOTICE`.

---

## Requesting more

Plenty exists that is not published here: full inference CSVs for every month, the complete figure archive with provenance sidecars, review session records, exploratory experiments, and the working repository's history. It is excluded to keep this release navigable, not because it is secret.

If you need something specific, open an issue. A pointed question is easier to answer well than an archive is to browse.
