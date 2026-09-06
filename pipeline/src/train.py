"""src/train.py
Pure PyTorch linear classifier training for the Perch-Hoplite pipeline.

Replaces the TensorFlow-based train_linear_classifier() from perch-hoplite.
Uses the same DataManager interface so DB/label logic is unchanged.

Performance note
----------------
Training examples are materialized once and the training tensors are copied to
the selected compute device before the optimization loop, rather than being
re-read from the database on every batch. Evaluation arrays stay on CPU because
evaluation is performed with NumPy after training.

On a GB10 this took a 256-step run from ~24 minutes (naive per-batch DB reads)
to ~16 seconds. Benchmark conditions -- hardware, label count, step count, and
the TensorFlow baseline it is compared against -- are in
docs/pytorch_port_summary.md. The speedup is workload- and hardware-dependent;
cite the conditions with the number.
"""
import logging

log = logging.getLogger(__name__)

# Warn when the persistent *training-data tensors alone* that will be copied to
# CUDA exceed 4 GiB. This is a byte threshold, not a label-count threshold.
# It intentionally excludes model/optimizer state, CUDA context, allocator
# caching, temporary activations, and other framework overhead.
_GPU_PRELOAD_WARN_BYTES = 4 * 1024**3


def _payload_nbytes(*arrays) -> int:
    """Return the exact NumPy payload size for non-None arrays."""
    return sum(int(a.nbytes) for a in arrays if a is not None)


def _format_bytes(num_bytes: int) -> str:
    """Human-readable binary size for log messages."""
    gib = num_bytes / 1024**3
    if gib >= 1.0:
        return f"{gib:.2f} GiB"
    return f"{num_bytes / 1024**2:.1f} MiB"


def torch_train_linear_classifier(
    data_manager,
    learning_rate: float,
    weak_neg_weight: float,
    num_train_steps: int,
    loss: str = "bce",
):
    """Train a linear classifier using PyTorch -- no TensorFlow required.

    Drop-in replacement for classifier_mod.train_linear_classifier().
    Uses the same DataManager interface so DB/label logic is unchanged.

    Parameters
    ----------
    data_manager : AgileDataManager
        Manages the DB, labels, weak-negative construction, and train/eval split.
    learning_rate : float
        Adam optimizer learning rate.
    weak_neg_weight : float
        Weight applied to weak (unlabeled) negatives in the loss.
    num_train_steps : int
        Number of gradient steps.
    loss : {"bce", "hinge"}
        Loss function. "bce" is numerically stable binary cross-entropy on
        logits; "hinge" uses labels mapped from {0, 1} to {-1, +1}.

    Returns
    -------
    (LinearClassifier, eval_scores)
        LinearClassifier matching the perch-hoplite saved format.
        eval_scores contains top1_acc, roc_auc, cmap, macro_f1,
        macro_f1_opt, and per_class_f1.

    Raises
    ------
    ValueError
        If loss is not "bce" or "hinge", or if the train/eval split contains
        no examples usable by this routine.

    Notes
    -----
    GPU memory logging reports the exact byte payload of the persistent
    training NumPy arrays that are copied to CUDA (embedding, multihot labels,
    and labeled-mask). It is not a prediction of total process GPU memory.
    """
    import numpy as np
    import torch
    import torch.nn as nn
    from tqdm import tqdm

    if loss not in {"bce", "hinge"}:
        raise ValueError(
            f"Unsupported loss {loss!r}; expected one of: 'bce', 'hinge'."
        )

    embedding_dim = data_manager.db.get_embedding_dim()
    target_labels = data_manager.get_target_labels()
    num_classes = len(target_labels)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    linear = nn.Linear(embedding_dim, num_classes, bias=True).to(device)
    nn.init.zeros_(linear.weight)
    nn.init.zeros_(linear.bias)

    optimizer = torch.optim.Adam(linear.parameters(), lr=learning_rate)

    def bce_loss_fn(logits, y_true, is_labeled):
        y = (
            y_true
            if isinstance(y_true, torch.Tensor)
            else torch.tensor(y_true, dtype=torch.float32, device=device)
        )
        m = (
            is_labeled
            if isinstance(is_labeled, torch.Tensor)
            else torch.tensor(is_labeled, dtype=torch.float32, device=device)
        )
        log_p = torch.nn.functional.logsigmoid(logits)
        log_not_p = torch.nn.functional.logsigmoid(-logits)
        raw_bce = -y * log_p - (1.0 - y) * log_not_p
        weights = (1.0 - m) * weak_neg_weight + m
        return (raw_bce * weights).mean()

    def hinge_loss_fn(logits, y_true, is_labeled):
        y_t = (
            y_true
            if isinstance(y_true, torch.Tensor)
            else torch.tensor(y_true, dtype=torch.float32, device=device)
        )
        y = 2 * y_t - 1
        m = (
            is_labeled
            if isinstance(is_labeled, torch.Tensor)
            else torch.tensor(is_labeled, dtype=torch.float32, device=device)
        )
        weights = (1.0 - m) * weak_neg_weight + m
        raw = torch.clamp(1.0 - y * logits, min=0.0)
        return (raw * weights).mean()

    loss_fn = {"bce": bce_loss_fn, "hinge": hinge_loss_fn}[loss]

    train_ids, eval_ids = data_manager.get_train_test_split()
    if len(train_ids) == 0:
        raise ValueError(
            "Training split is empty; at least one training ID is required. "
            "Check the train/eval split and available labels."
        )
    if len(eval_ids) == 0:
        raise ValueError(
            "Evaluation split is empty; this routine returns held-out metrics "
            "and therefore requires at least one evaluation ID."
        )

    def _load_ids_to_arrays(ids, add_weak_negatives):
        batches = list(
            data_manager.batched_example_iterator(
                ids, add_weak_negatives=add_weak_negatives, repeat=False
            )
        )
        if not batches:
            return None, None, None, None
        emb = np.concatenate([b.embedding for b in batches], axis=0).astype(
            np.float32, copy=False
        )
        mh = np.concatenate([b.multihot for b in batches], axis=0).astype(
            np.float32, copy=False
        )
        ilm = np.concatenate([b.is_labeled_mask for b in batches], axis=0).astype(
            np.float32, copy=False
        )
        idxs = np.concatenate([b.idx for b in batches], axis=0)
        return emb, mh, ilm, idxs

    # Load training data once. add_weak_negatives=True may change the number of
    # materialized training rows, so memory is measured from the actual arrays,
    # not inferred from len(train_ids).
    #
    # This read is the slowest phase of training on a large database -- it can
    # take several minutes while the GPU sits idle -- so announce it before
    # starting. The measured payload is logged afterwards, once the arrays exist.
    log.info(
        "Materializing training examples from %d train IDs "
        "(add_weak_negatives=True)...",
        len(train_ids),
    )
    train_emb_np, train_mh_np, train_ilm_np, _ = _load_ids_to_arrays(
        train_ids, True
    )
    if train_emb_np is None or train_emb_np.shape[0] == 0:
        raise ValueError(
            "Training IDs were present, but the DataManager produced no training "
            "examples. Check label filtering and weak-negative generation."
        )

    n_train = train_emb_np.shape[0]
    preload_bytes = _payload_nbytes(train_emb_np, train_mh_np, train_ilm_np)
    preload_size = _format_bytes(preload_bytes)

    if device.type == "cuda" and preload_bytes > _GPU_PRELOAD_WARN_BYTES:
        log.warning(
            "Training-data preload is %s for %d materialized examples from "
            "%d train IDs (%.2fx expansion, %d-D embeddings). This exceeds the "
            "4 GiB warning threshold. The value covers embedding/label/mask "
            "tensors only; total CUDA memory will be higher because model, "
            "Adam state, activations, CUDA context, and allocator overhead are "
            "not included.",
            preload_size,
            n_train,
            len(train_ids),
            n_train / len(train_ids),
            embedding_dim,
        )
    else:
        log.info(
            "Pre-loading %d materialized training examples from %d train IDs "
            "(%.2fx expansion, %s payload) onto %s...",
            n_train,
            len(train_ids),
            n_train / len(train_ids),
            preload_size,
            device,
        )

    train_emb = torch.as_tensor(train_emb_np, dtype=torch.float32, device=device)
    train_mh = torch.as_tensor(train_mh_np, dtype=torch.float32, device=device)
    train_ilm = torch.as_tensor(train_ilm_np, dtype=torch.float32, device=device)

    # Evaluation is performed below with NumPy, so keep evaluation data on CPU.
    eval_emb, eval_mh, _eval_ilm, eval_idxs = _load_ids_to_arrays(eval_ids, False)
    if eval_emb is None or eval_emb.shape[0] == 0:
        raise ValueError(
            "Evaluation IDs were present, but the DataManager produced no "
            "evaluation examples. Check label filtering and split construction."
        )

    log.info(
        "Loaded %d train examples (from %d train IDs) onto %s and kept %d eval "
        "examples (from %d eval IDs) on CPU",
        n_train,
        len(train_ids),
        device,
        eval_emb.shape[0],
        len(eval_ids),
    )

    # Training loop -- pure in-memory mini-batches.
    linear.train()
    rng = np.random.default_rng(seed=42)
    batch_size = min(512, n_train)

    with tqdm(total=num_train_steps, desc="Training") as pbar:
        for step in range(num_train_steps):
            idx = torch.tensor(
                rng.choice(n_train, size=batch_size, replace=False), device=device
            )
            logits = linear(train_emb[idx])
            loss_val = loss_fn(logits, train_mh[idx], train_ilm[idx])
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            if step % 32 == 0:
                pbar.set_postfix({"Loss": f"{loss_val.item():.8f}"})
            pbar.update(1)

    # Extract weights in perch-hoplite's expected orientation.
    linear.eval()
    with torch.no_grad():
        beta = linear.weight.T.cpu().numpy()  # (embedding_dim, num_classes)
        beta_bias = linear.bias.cpu().numpy()  # (num_classes,)

    # Evaluate using the existing perch-hoplite metrics implementation.
    from perch_hoplite.agile import classifier as _clf_mod
    from perch_hoplite.agile import metrics as _metrics
    from ml_collections import config_dict as _cd

    pred_logits = np.dot(eval_emb, beta) + beta_bias
    true_labels = eval_mh

    labeled = np.where(true_labels.sum(axis=1) > 0)
    if labeled[0].size == 0:
        raise ValueError(
            "Evaluation data contains no positive labels; top-1 accuracy and "
            "the requested held-out metrics are not meaningful."
        )

    top_preds = np.argmax(pred_logits, axis=1)
    top1 = true_labels[np.arange(top_preds.shape[0]), top_preds][labeled].mean()
    rocs = _metrics.roc_auc(
        logits=pred_logits, labels=true_labels, sample_threshold=1
    )
    cmaps = _metrics.cmap(logits=pred_logits, labels=true_labels, sample_threshold=1)

    # Per-class F1 on the same held-out eval split used for cMAP/ROC AUC.
    from src.f1_metrics import per_class_f1

    _f1 = per_class_f1(pred_logits, true_labels, target_labels)

    eval_scores = {
        "top1_acc": float(top1),
        "roc_auc": float(rocs["macro"]),
        "cmap": float(cmaps["macro"]),
        "macro_f1": _f1["macro_f1_at_0"],
        "macro_f1_opt": _f1["macro_f1_opt"],
        "per_class_f1": _f1,
    }

    _emb_cfg = _cd.ConfigDict()
    lin_cls = _clf_mod.LinearClassifier(
        beta=beta,
        beta_bias=beta_bias,
        classes=target_labels,
        embedding_model_config=_emb_cfg,
    )
    return lin_cls, eval_scores
