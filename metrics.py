"""Detection metrics, event-level view and uncertainty intervals."""

import math

import numpy as np
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix,
    precision_recall_fscore_support, roc_auc_score,
)

from .windowing import contiguous_runs


def safe_auc(y, s):
    y = np.asarray(y).astype(int)
    return float(roc_auc_score(y, s)) if len(np.unique(y)) >= 2 else np.nan


def safe_ap(y, s):
    y = np.asarray(y).astype(int)
    return float(average_precision_score(y, s)) if len(np.unique(y)) >= 2 else np.nan


def compute_metrics(y_true, y_pred, scores=None):
    """Window-level confusion matrix, rates and (optionally) ranking metrics."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    out = {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1,
           "FPR": fp / (fp + tn + 1e-12), "TN": tn, "FP": fp, "FN": fn, "TP": tp}
    if scores is not None:
        out["ROC-AUC"] = safe_auc(y_true, scores)
        out["PR-AUC"] = safe_ap(y_true, scores)
    return out


def event_metrics(starts, y, y_pred, stride):
    """Event-level (episode-level) view: an attack event is counted as
    detected if at least one window inside it raises an alarm. Window-level
    recall penalises a detector that flags an attack once and then settles,
    even though an operator has already been alerted; event recall is the
    quantity that matters operationally. Reported as a supplementary metric
    alongside, never instead of, the window-level numbers."""
    y = np.asarray(y).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    order = np.argsort(starts, kind="stable")
    st, yy, pp = np.asarray(starts)[order], y[order], y_pred[order]
    n_ev = n_hit = 0
    for g in contiguous_runs(st, stride):
        v, p = yy[g], pp[g]
        d = np.flatnonzero(np.diff(np.r_[0, v, 0]))
        for a, b in zip(d[0::2], d[1::2]):
            n_ev += 1
            if p[a:b].any():
                n_hit += 1
    return {"EventCount": n_ev,
            "EventRecall": (n_hit / n_ev) if n_ev else float("nan"),
            "EventsDetected": n_hit}


def wilson_ci(successes, n, z=1.96):
    """95% Wilson score interval for precision (TP, TP+FP) and recall
    (TP, TP+FN). A rate estimated from a handful of positives is meaningless
    without an interval."""
    if n == 0:
        return (np.nan, np.nan)
    phat = successes / n
    denom = 1 + z ** 2 / n
    centre = (phat + z ** 2 / (2 * n)) / denom
    half = (z / denom) * math.sqrt(phat * (1 - phat) / n + z ** 2 / (4 * n ** 2))
    return (max(0.0, centre - half), min(1.0, centre + half))


def bootstrap_ci(y, s, fn, n_boot, seed):
    """Percentile bootstrap 95% interval for ROC-AUC / PR-AUC."""
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=float)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(fn(y[idx], s[idx]))
    if not vals:
        return (np.nan, np.nan)
    return tuple(np.percentile(vals, [2.5, 97.5]))
