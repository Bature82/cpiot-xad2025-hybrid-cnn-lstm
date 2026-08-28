"""Anomaly evidence, score normalisation, smoothing and thresholds.

All reference statistics come from the calibration holdout only: residuals on
data a model has fitted are optimistically small, so calibrating on unseen
normals is what makes a validation operating point survive the move to test.
"""

import numpy as np

from .windowing import contiguous_runs


def topk_mean(x2d, k):
    """Mean of the k largest per-feature residuals in each row."""
    x2d = np.asarray(x2d, dtype=np.float32)
    if x2d.ndim != 2 or x2d.shape[1] == 0:
        return np.zeros((x2d.shape[0],), dtype=np.float32)
    D = x2d.shape[1]
    kk = int(min(max(1, k), D))
    part = np.partition(x2d, D - kk, axis=1)[:, D - kk:]
    return part.mean(axis=1)


def fit_residual_stats(pred, true, eps):
    """Per-feature median and MAD of the absolute forecast residual, estimated
    on calibration NORMAL windows only."""
    r = np.abs(pred - true)
    med = np.median(r, axis=0, keepdims=True)
    mad = np.median(np.abs(r - med), axis=0, keepdims=True) * 1.4826 + eps
    return med.astype(np.float32), mad.astype(np.float32)


def residual_matrix(pred, true, med, mad):
    """Robustly normalised per-feature residual evidence, computed once per
    split and reused for every top-k in the selection grid."""
    r = np.abs(pred - true)
    return np.log1p(np.abs((r - med) / mad)).astype(np.float32)


def fit_mad(scores):
    """Median and scaled MAD of a score distribution (the standardisation
    constants for one stream)."""
    scores = np.asarray(scores, dtype=float)
    mu = np.median(scores)
    mad = np.median(np.abs(scores - mu)) * 1.4826 + 1e-9
    return float(mu), float(mad)


def apply_mad(scores, mu, mad):
    """Robust z-score, with non-finite entries pinned to the centre."""
    scores = np.asarray(scores, dtype=float)
    scores = np.where(np.isfinite(scores), scores, mu)
    return (scores - mu) / (mad + 1e-9)


def smooth_scores(starts, scores, stride, k):
    """Trailing moving average of length k over temporally adjacent windows.
    Physical-process attacks persist for many consecutive samples while
    reconstruction noise does not, so averaging along the window sequence
    raises the signal-to-noise ratio of the anomaly score. The average is
    causal (past and present only), so it remains deployable online, and it
    is applied independently within each contiguous run of windows."""
    scores = np.asarray(scores, dtype=np.float64)
    if k is None or k <= 1 or scores.size == 0:
        return scores
    order = np.argsort(starts, kind="stable")
    inv = np.empty_like(order)
    inv[order] = np.arange(order.size)
    s_sorted = scores[order]
    out = np.empty_like(s_sorted)
    for g in contiguous_runs(np.asarray(starts)[order], stride):
        v = s_sorted[g]
        c = np.cumsum(np.r_[0.0, v])
        idx = np.arange(v.size)
        lo = np.maximum(0, idx - k + 1)
        out[g] = (c[idx + 1] - c[lo]) / (idx + 1 - lo)
    return out[inv]


def threshold_from_normals(scores, target_fpr):
    """Upper quantile of the pooled NORMAL score distribution (calibration
    holdout + validation normals). Pooling gives a far larger normal sample
    than validation alone, which is what stabilises the realised false
    positive rate on unseen data. Nudged by one ULP so that ties on the
    quantile value are resolved as negatives."""
    s = np.asarray(scores, dtype=float)
    s = s[np.isfinite(s)]
    if s.size < 20:
        return float(np.nextafter(
            float(np.percentile(s, 99.5)) if s.size else 0.0, np.inf))
    thr = float(np.percentile(s, 100.0 * (1.0 - target_fpr)))
    return float(np.nextafter(thr, np.inf))


def best_threshold(y, scores, metric="F1", n_grid=512):
    """Threshold maximising `metric` on the VALIDATION split. Reported
    alongside the FPR-constrained threshold: they answer different deployment
    questions (a bounded alarm budget, the best balanced detection, or the
    highest overall correctness). Whichever is quoted, it was chosen without
    ever looking at a test label."""
    y = np.asarray(y).astype(int)
    s = np.asarray(scores, dtype=float)
    if len(np.unique(y)) < 2 or s.size == 0:
        return float("inf"), 0.0
    qs = np.unique(np.percentile(s, np.linspace(0.1, 99.9, n_grid)))
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    best_t, best_v = float(np.max(s)) + 1.0, -1.0
    for t in qs:
        pred = (s > t).astype(int)
        tp = int(np.sum((pred == 1) & (y == 1)))
        fp = int(np.sum((pred == 1) & (y == 0)))
        fn = n_pos - tp
        tn = n_neg - fp
        if metric == "Accuracy":
            v = (tp + tn) / max(1, n_pos + n_neg)
        else:
            v = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        if v > best_v:
            best_v, best_t = v, float(t)
    return float(np.nextafter(best_t, np.inf)), best_v


def hysteresis_predict(starts, scores, stride, thr_hi, thr_lo):
    """Two-threshold (Schmitt-trigger) alarm. An alarm is raised when the score
    rises above thr_hi and is held until the score falls back below thr_lo,
    independently within each contiguous run of windows.

    A single threshold implicitly assumes each window is an independent
    decision. Attacks on a physical plant are not independent events: the
    evidence is strong at onset, weaker while the process settles into the
    manipulated state, and strong again at recovery. A single threshold
    therefore fragments one attack into several short detections and misses its
    quiet middle. Holding the alarm through the dip costs almost no extra false
    positives, because on normal data the score rarely crosses thr_hi in the
    first place."""
    s = np.asarray(scores, dtype=float)
    if s.size == 0:
        return np.zeros(0, dtype=int)
    if not np.isfinite(thr_lo) or thr_lo >= thr_hi:
        return (s > thr_hi).astype(int)
    order = np.argsort(starts, kind="stable")
    inv = np.empty_like(order)
    inv[order] = np.arange(order.size)
    ss = s[order]
    out = np.zeros(ss.size, dtype=np.int8)
    for g in contiguous_runs(np.asarray(starts)[order], stride):
        on = False
        for t in g:
            if on:
                if ss[t] <= thr_lo:
                    on = False
            elif ss[t] > thr_hi:
                on = True
            out[t] = 1 if on else 0
    return out[inv].astype(int)
