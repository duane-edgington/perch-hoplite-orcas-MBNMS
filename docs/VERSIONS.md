# VERSIONS.md — pinned environment

The exact tool versions used to produce the released results. Resampling output in particular can differ across SoX versions and builds, so these are part of the reproducibility record rather than incidental detail.

All package versions below were read from the production venv on the primary host with `importlib.metadata.version`, not from memory or from a requirements file.

---

## Resampling (stage 2)

| Component | Version | How to check |
|---|---|---|
| **SoX** | **14.4.2** (`/usr/bin/sox`) | `sox --version` |
| OS | Ubuntu 24.04.3 LTS | `lsb_release -d` or `cat /etc/os-release` |
| Kernel / arch | Linux aarch64 | `uname -srm` |

SoX flags — these define the output bytes and must not be varied:

```
sox <in> -b 16 <out> rate -v 32000 highpass 10 fade 0.1 -0 0.1 vol 3
```

See [../resampling/README.md](../resampling/README.md) for what each flag does and why `vol 3` is not optional.

---

## Embedding (stage 3)

| Component | Version | How to check |
|---|---|---|
| Python | 3.12.3 | `python3 --version` |
| PyTorch | 2.12.1+cu130 | `python3 -c "import torch; print(torch.__version__)"` |
| CUDA | 13.0 | `nvcc --version` |
| GPU | NVIDIA GB10 (Blackwell, compute capability 12.1) | `nvidia-smi` |
| PyTorch Perch V2 port | [`perch2-pytorch-port`](https://github.com/duane-edgington/perch2-pytorch-port) | `git -C <clone> rev-parse HEAD` |
| Perch V2 weights | ONNX-extracted `weights.npz` + `graph_manifest.json`, regenerated locally by `extract_weights.py` | see the port repo's README |

Perch V2 weights are **not redistributed** by either repository. The port repo's
`extract_weights.py` regenerates them from Google's published model; it needs only
`onnx`, `huggingface_hub`, and `numpy`, and runs on CPU with no GPU and no TensorFlow.

Embedding runs used `--device cuda --compile`, hop size 5.0 s, batch size 8. Per-window peak normalization to 0.25 is applied inside the adapter and is not configurable.

A second GPU host was validated as producing identical review rendering and results against the same databases, with slightly different package versions (torch 2.13.0, numpy 2.5.2, usearch 2.26.0, gradio 6.25.0). Those differences appear low-risk but were not stress-tested beyond one session. The numbers in this release come from the primary host.

---

## Classification, inference, review (stage 5)

| Component | Version |
|---|---|
| perch-hoplite | 1.0.2 |
| usearch | 2.25.3 |
| torch | 2.12.1+cu130 |
| numpy | 2.4.4 |
| scipy | 1.18.0 |
| scikit-learn | 1.9.0 |
| soundfile | 0.14.0 |
| librosa | 0.11.0 |
| ml-collections | 0.1.1 |
| gradio | 6.15.1 (6.25.0 also validated) |
| soxr | 1.1.0 |
| timm | 1.0.27 |

**perch-hoplite is installed from PyPI at the pin above** — `pip install perch-hoplite==1.0.2`, never `perch-hoplite[tf]` and not from a git checkout. 1.0.2 is the version the released results were produced with.

**No TensorFlow is required.** The pipeline runs pure PyTorch end to end; `phase2_classify.py` injects a TF mock to satisfy upstream perch-hoplite imports. TensorFlow is only an optional `[tf]` extra of perch-hoplite, so a plain install is TF-free by construction. If you find yourself installing TensorFlow to run this, something has gone wrong — please open an issue.

Training used `--train-ratio 0.8` and seed 42 for `orca_v10`. The step count is recorded in `../models/orca_v10.metrics.json`, which is the authority; cite that file rather than this line.

---

## Reproducing the environment

From the repository root:

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

That is the whole install. `requirements.txt` pins `perch-hoplite==1.0.2`, which brings `usearch`, `librosa`, `matplotlib`, `ml-collections`, `numpy`, `scipy`, and `pandas` with it.

On a host needing a specific CUDA build, install torch **first** from the matching index, then run the line above — the requirement is already satisfied and pip will not replace it:

```bash
# NVIDIA GB10 / CUDA 13 (the primary host):
pip install torch --index-url https://download.pytorch.org/whl/cu130
```

For embedding, additionally clone [`perch2-pytorch-port`](https://github.com/duane-edgington/perch2-pytorch-port) and run its `extract_weights.py` — see [REPRODUCE.md](REPRODUCE.md) stage 0.

---

## Full pip freeze

The table above is the summary record. For the belt-and-braces version, commit the
output of `pip freeze` from the production venv as `environment/pip-freeze.txt` and
reference it here.
