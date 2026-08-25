"""src/f1_metrics.py
Per-class precision / recall / F1 for the Perch-Hoplite classifier eval split.

Designed to be called from torch_train_linear_classifier()'s eval block, on the
SAME (pred_logits, true_labels) the existing roc_auc / cmap metrics use — so the
F1 numbers line up one-to-one with the cmap / ROC-AUC you already report.

Multi-label: each class is scored independently against its logit column.
Reports F1 at both a fixed threshold (logit >= 0.0, matching --logit-threshold 0.0)
and the F1-optimal per-class threshold from a full score sweep.

Dependency: numpy only. Validated by the assertions in _selftest() below.
"""
import numpy as np


def _prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f


def metrics_at_threshold(y_true, score, thr):
    """Precision/recall/F1 for one class at a fixed decision threshold (score >= thr)."""
    y_true = np.asarray(y_true)
    score = np.asarray(score)
    pred = score >= thr
    tp = int(np.sum(pred & (y_true == 1)))
    fp = int(np.sum(pred & (y_true == 0)))
    fn = int(np.sum((~pred) & (y_true == 1)))
    p, r, f = _prf(tp, fp, fn)
    return {"threshold": float(thr), "tp": tp, "fp": fp, "fn": fn,
            "precision": p, "recall": r, "f1": f}


def best_f1_threshold(y_true, score):
    """Threshold maximizing F1 for one class, via a full score sweep.

    Evaluates every predict-top-k-positive prefix, then re-derives metrics at the
    concrete threshold so tp/fp/fn are tie-consistent with the reported threshold.
    """
    y_true = np.asarray(y_true)
    score = np.asarray(score)
    if int(np.sum(y_true == 1)) == 0:
        return {"threshold": float("inf"), "tp": 0, "fp": 0, "fn": 0,
                "precision": 0.0, "recall": 0.0, "f1": 0.0}
    order = np.argsort(-score, kind="mergesort")
    ys = y_true[order]
    ss = score[order]
    P = int(np.sum(y_true == 1))
    tp = np.cumsum(ys == 1).astype(float)
    fp = np.cumsum(ys == 0).astype(float)
    fn = P - tp
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        rec = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
    best = int(np.argmax(f1))
    return metrics_at_threshold(y_true, score, float(ss[best]))


def per_class_f1(pred_logits, true_labels, class_names):
    """Per-class + macro F1 for the eval split. JSON-serializable (fold into eval_scores).

    Parameters
    ----------
    pred_logits : array [N, C]   classifier logits on the eval set
    true_labels : array [N, C]   multi-hot ground truth on the eval set
    class_names : list[str]      length C, column order (data_manager.get_target_labels())

    Macro F1 averages only over classes with >=1 positive in the eval split, matching
    cmap/roc_auc's sample_threshold=1 behavior. Classes with 0 support are still listed
    (f1=0.0, support=0) so gaps are visible.
    """
    pred_logits = np.asarray(pred_logits)
    true_labels = np.asarray(true_labels)
    assert pred_logits.shape == true_labels.shape, \
        f"shape mismatch: logits {pred_logits.shape} vs labels {true_labels.shape}"
    assert pred_logits.shape[1] == len(class_names), \
        f"{pred_logits.shape[1]} columns but {len(class_names)} class names"

    per_class = {}
    f0_scored, fo_scored = [], []
    for j, name in enumerate(class_names):
        yt = true_labels[:, j]
        sc = pred_logits[:, j]
        support = int(np.sum(yt == 1))
        at0 = metrics_at_threshold(yt, sc, 0.0)
        opt = best_f1_threshold(yt, sc)
        per_class[name] = {
            "support": support,
            "f1_at_0": at0["f1"],
            "precision_at_0": at0["precision"],
            "recall_at_0": at0["recall"],
            "f1_opt": opt["f1"],
            "precision_opt": opt["precision"],
            "recall_opt": opt["recall"],
            "opt_threshold": (opt["threshold"] if support > 0 else None),
        }
        if support > 0:
            f0_scored.append(at0["f1"])
            fo_scored.append(opt["f1"])
    return {
        "n_eval": int(true_labels.shape[0]),
        "per_class": per_class,
        "macro_f1_at_0": float(np.mean(f0_scored)) if f0_scored else 0.0,
        "macro_f1_opt": float(np.mean(fo_scored)) if fo_scored else 0.0,
        "macro_over_n_classes": len(f0_scored),
    }


def _selftest():
    rng = np.random.default_rng(0)
    N, C = 4000, 5
    names = ["orca_call", "humpback_song", "dolphin_call", "ship_noise", "other"]
    Y = np.zeros((N, C), dtype=int)
    S = rng.normal(-2.0, 1.0, size=(N, C))
    for j, (sup, sep) in enumerate(zip([400, 300, 40, 60, 200], [3.5, 3.0, 2.2, 2.0, 1.5])):
        pos = rng.choice(N, size=sup, replace=False)
        Y[pos, j] = 1
        S[pos, j] = rng.normal(sep, 1.0, size=sup)
    rep = per_class_f1(S, Y, names)
    ok = True
    for name in names:
        c = rep["per_class"][name]
        if c["f1_opt"] + 1e-9 < c["f1_at_0"]:
            print(f"FAIL {name}: opt < fixed"); ok = False
    # perfectly separable -> 1.0 ; empty -> 0.0
    if abs(best_f1_threshold(np.array([0, 0, 1, 1]),
                             np.array([-5., -4., 4., 5.]))["f1"] - 1.0) > 1e-9:
        print("FAIL: separable != 1.0"); ok = False
    if best_f1_threshold(np.zeros(10, int), rng.normal(size=10))["f1"] != 0.0:
        print("FAIL: empty != 0.0"); ok = False
    print(json.dumps(rep, indent=2))
    print("SELFTEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import json
    raise SystemExit(_selftest())
