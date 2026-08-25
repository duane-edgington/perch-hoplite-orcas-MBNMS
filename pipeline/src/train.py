"""src/train.py
Pure PyTorch linear classifier training for the Perch-Hoplite pipeline.

Replaces the TensorFlow-based train_linear_classifier() from perch-hoplite.
Uses the same DataManager interface so all DB/label logic is unchanged.
Training is ~37x faster than TF by pre-loading all labeled embeddings into
GPU memory at startup rather than reading them per-batch from the DB.
"""
import logging

log = logging.getLogger(__name__)

# Memory safeguard threshold: warn if label count would use > ~4GB GPU memory
_LABEL_WARN_THRESHOLD = 50_000


def torch_train_linear_classifier(
    data_manager,
    learning_rate: float,
    weak_neg_weight: float,
    num_train_steps: int,
    loss: str = "bce",
):
    """Train a linear classifier using PyTorch — no TensorFlow required.

    Drop-in replacement for classifier_mod.train_linear_classifier().
    Uses the same DataManager interface so all DB/label logic is unchanged.

    Parameters
    ----------
    data_manager : AgileDataManager
        Manages the DB, labels, and train/eval split.
    learning_rate : float
        Adam optimizer learning rate.
    weak_neg_weight : float
        Weight applied to weak (unlabeled) negatives in the loss.
    num_train_steps : int
        Number of gradient steps.
    loss : str
        "bce" (binary cross-entropy, default) or "hinge".

    Returns
    -------
    (LinearClassifier, eval_scores)
        LinearClassifier matching the perch-hoplite saved format.
        eval_scores dict with keys: top1_acc, roc_auc, cmap.
    """
    import numpy as np
    import torch
    import torch.nn as nn
    from tqdm import tqdm

    embedding_dim = data_manager.db.get_embedding_dim()
    target_labels = data_manager.get_target_labels()
    num_classes   = len(target_labels)
    device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    linear = nn.Linear(embedding_dim, num_classes, bias=True).to(device)
    nn.init.zeros_(linear.weight)
    nn.init.zeros_(linear.bias)

    optimizer = torch.optim.Adam(linear.parameters(), lr=learning_rate)

    def bce_loss_fn(logits, y_true, is_labeled):
        y = (y_true if isinstance(y_true, torch.Tensor)
             else torch.tensor(y_true, dtype=torch.float32, device=device))
        m = (is_labeled if isinstance(is_labeled, torch.Tensor)
             else torch.tensor(is_labeled, dtype=torch.float32, device=device))
        log_p     = torch.nn.functional.logsigmoid(logits)
        log_not_p = torch.nn.functional.logsigmoid(-logits)
        raw_bce   = -y * log_p - (1.0 - y) * log_not_p
        weights   = (1.0 - m) * weak_neg_weight + m
        return (raw_bce * weights).mean()

    def hinge_loss_fn(logits, y_true, is_labeled):
        y_t = (y_true if isinstance(y_true, torch.Tensor)
               else torch.tensor(y_true, dtype=torch.float32, device=device))
        y = 2 * y_t - 1
        m = (is_labeled if isinstance(is_labeled, torch.Tensor)
             else torch.tensor(is_labeled, dtype=torch.float32, device=device))
        weights = (1.0 - m) * weak_neg_weight + m
        raw = torch.clamp(1.0 - y * logits, min=0.0)
        return (raw * weights).mean()

    loss_fn = hinge_loss_fn if loss == "hinge" else bce_loss_fn

    train_ids, eval_ids = data_manager.get_train_test_split()

    # ── Pre-load ALL embeddings + labels into GPU memory ─────────────────────
    # Much faster than per-batch DB reads (17280 × 1536 float32 ≈ 100 MB).
    # Memory scales with LABEL COUNT only, not DB size — safe for multi-month DBs.
    _n_labels = len(train_ids) + len(eval_ids)
    _est_gb   = _n_labels * embedding_dim * 4 / 1e9
    if _n_labels > _LABEL_WARN_THRESHOLD:
        log.warning(
            "Large label set: %d examples × %d dims = %.1f GB GPU memory. "
            "Consider reducing --train-ratio or using --batch-size.",
            _n_labels, embedding_dim, _est_gb,
        )
    else:
        log.info("Pre-loading %d labeled examples (%.1f MB) into %s...",
                 _n_labels, _est_gb * 1000, device)

    def _load_ids_to_tensors(ids, add_weak_negatives):
        batches = list(data_manager.batched_example_iterator(
            ids, add_weak_negatives=add_weak_negatives, repeat=False))
        if not batches:
            return None, None, None, None
        emb  = np.concatenate([b.embedding      for b in batches], axis=0)
        mh   = np.concatenate([b.multihot        for b in batches], axis=0)
        ilm  = np.concatenate([b.is_labeled_mask for b in batches], axis=0)
        idxs = np.concatenate([b.idx             for b in batches], axis=0)
        return (
            torch.tensor(emb,  dtype=torch.float32, device=device),
            torch.tensor(mh,   dtype=torch.float32, device=device),
            torch.tensor(ilm,  dtype=torch.float32, device=device),
            idxs,
        )

    train_emb, train_mh, train_ilm, _         = _load_ids_to_tensors(train_ids, True)
    eval_emb,  eval_mh,  eval_ilm,  eval_idxs = _load_ids_to_tensors(eval_ids,  False)
    n_train = train_emb.shape[0] if train_emb is not None else 0
    log.info("Loaded %d train + %d eval examples onto %s",
             n_train, eval_emb.shape[0] if eval_emb is not None else 0, device)

    # ── Training loop — pure in-memory mini-batches ───────────────────────────
    linear.train()
    rng        = np.random.default_rng(seed=42)
    batch_size = min(512, n_train)

    with tqdm(total=num_train_steps, desc="Training") as pbar:
        for step in range(num_train_steps):
            idx    = torch.tensor(
                rng.choice(n_train, size=batch_size, replace=False), device=device)
            logits    = linear(train_emb[idx])
            loss_val  = loss_fn(logits, train_mh[idx], train_ilm[idx])
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            if step % 32 == 0:
                pbar.set_postfix({"Loss": f"{loss_val.item():.8f}"})
            pbar.update(1)

    # ── Extract weights ───────────────────────────────────────────────────────
    linear.eval()
    with torch.no_grad():
        beta      = linear.weight.T.cpu().numpy()   # (embedding_dim, num_classes)
        beta_bias = linear.bias.cpu().numpy()        # (num_classes,)

    # ── Evaluate ─────────────────────────────────────────────────────────────
    from perch_hoplite.agile import classifier as _clf_mod
    from perch_hoplite.agile import metrics    as _metrics
    from ml_collections import config_dict     as _cd

    pred_logits = np.dot(eval_emb.cpu().numpy(), beta) + beta_bias
    true_labels = eval_mh.cpu().numpy()

    labeled  = np.where(true_labels.sum(axis=1) > 0)
    top_preds = np.argmax(pred_logits, axis=1)
    top1      = true_labels[np.arange(top_preds.shape[0]), top_preds][labeled].mean()
    rocs      = _metrics.roc_auc(logits=pred_logits, labels=true_labels, sample_threshold=1)
    cmaps     = _metrics.cmap(logits=pred_logits, labels=true_labels, sample_threshold=1)

    # Per-class F1 on the SAME held-out eval split cmap/roc_auc use, so the numbers
    # line up one-to-one with the cmap column. See src/f1_metrics.py.
    from src.f1_metrics import per_class_f1
    _f1 = per_class_f1(pred_logits, true_labels, target_labels)

    eval_scores = {
        "top1_acc":     float(top1),
        "roc_auc":      float(rocs["macro"]),
        "cmap":         float(cmaps["macro"]),
        "macro_f1":     _f1["macro_f1_at_0"],
        "macro_f1_opt": _f1["macro_f1_opt"],
        "per_class_f1": _f1,
    }

    _emb_cfg = _cd.ConfigDict()
    lin_cls  = _clf_mod.LinearClassifier(
        beta=beta,
        beta_bias=beta_bias,
        classes=target_labels,
        embedding_model_config=_emb_cfg,
    )
    return lin_cls, eval_scores
