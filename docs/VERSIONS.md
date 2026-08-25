# VERSIONS.md — pinned environment

The exact tool versions used to produce the released results. Resampling output in particular can differ across SoX versions and builds, so these are part of the reproducibility record rather than incidental detail.

> **Maintainer note — verify before release.** Entries marked `TO VERIFY` must be filled in from the actual production host. Run the commands shown and paste the output. Anything left as `TO VERIFY` in a published release is worse than omitting the section.

---

## Resampling (stage 2)

| Component | Version | How to check |
|---|---|---|
| **SoX** | **14.4.2** (`/usr/bin/sox`) | `sox --version` |
| OS | `TO VERIFY` | `lsb_release -d` or `cat /etc/os-release` |

SoX flags — these define the output bytes and must not be varied:

```
sox <in> -b 16 <out> rate -v 32000 highpass 10 fade 0.1 -0 0.1 vol 3
```

See [../resampling/README.md](../resampling/README.md) for what each flag does and why `vol 3` is not optional.

---

## Embedding (stage 3)

| Component | Version | How to check |
|---|---|---|
| Python | `TO VERIFY` (3.12.x) | `python3 --version` |
| PyTorch | 2.12.1+cu130 | `python3 -c "import torch; print(torch.__version__)"` |
| CUDA | 13.0 | `nvcc --version` |
| GPU | NVIDIA GB10 (Blackwell, compute capability 12.1) | `nvidia-smi` |
| perch-pytorch | `TO VERIFY` (commit hash) | `git -C ~/perch-pytorch rev-parse HEAD` |
| Perch V2 weights | ONNX-extracted, `weights.npz` + `graph_manifest.json` | see perch-pytorch |

Embedding runs used `--device cuda --compile`, hop size 5.0 s, batch size 8. Per-window peak normalization to 0.25 is applied inside the adapter and is not configurable.

A second GPU host was validated as producing identical review rendering and results against the same databases, with slightly different package versions (torch 2.13.0, numpy 2.5.2, usearch 2.26.0, gradio 6.25.0). Those differences appear low-risk but were not stress-tested beyond one session. The numbers in this release come from the primary host.

---

## Classification, inference, review (stages 5)

| Component | Version |
|---|---|
| perch-hoplite | 1.0.1 |
| torch | 2.12.1+cu130 |
| gradio | 6.15.1 (6.25.0 also validated) |
| soundfile | 0.13.1 |
| librosa | 0.11.0 |
| numpy | `TO VERIFY` |
| scipy, scikit-learn, matplotlib, pandas | see `../requirements.txt` |

**No TensorFlow is required.** The pipeline runs pure PyTorch end to end; `phase2_classify.py` injects a TF mock to satisfy upstream perch-hoplite imports. If you find yourself installing TensorFlow to run this, something has gone wrong — please open an issue.

Training used `--num-steps 512 --train-ratio 0.8`, seed 42, for `orca_v10`.

---

## Reproducing the environment

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r ../requirements.txt
pip install git+https://github.com/google-research/perch-hoplite.git
```

Then, for embedding only, the separate perch-pytorch environment — see [REPRODUCE.md](REPRODUCE.md) stage 0.

---

## Full pip freeze

> `TO VERIFY` — paste the output of `pip freeze` from the production venv here at release, or commit it as `environment/pip-freeze.txt`. This is the belt-and-braces record for anyone who cannot reproduce results from the summary table above.
