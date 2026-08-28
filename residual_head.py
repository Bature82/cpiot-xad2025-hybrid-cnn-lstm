"""The residual-space classifier behind the "semisup" and "hybrid" scorers.

The forecasting network is trained on normal data only and never sees an
anomaly. What it produces is a per-feature residual vector per window. The
unsupervised scorer collapses that vector with a single hand-chosen statistic
(a top-k mean). A classifier fitted on the SAME residual vectors, using only
the train split's own labels, learns which combination of residuals actually
distinguishes an attack - information the corpus already contains and the
hand-chosen statistic discards.
"""

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import CFG
from .metrics import safe_ap
from .scoring import topk_mean


def residual_features(R):
    """Summarise a (n_windows, n_features) normalised residual matrix into a
    fixed-width per-window descriptor. Order statistics rather than raw
    per-feature residuals, so the descriptor has the same meaning regardless of
    how many features a stream contributes and cannot memorise feature
    identity."""
    R = np.asarray(R, dtype=np.float32)
    if R.ndim != 2 or R.shape[0] == 0 or R.shape[1] == 0:
        return np.zeros((R.shape[0] if R.ndim == 2 else 0, 0), dtype=np.float32)
    qs = np.percentile(R, [10, 25, 50, 75, 90, 95, 99], axis=1).T.astype(np.float32)
    parts = [R.mean(axis=1, keepdims=True).astype(np.float32),
             R.std(axis=1, keepdims=True).astype(np.float32),
             R.max(axis=1, keepdims=True).astype(np.float32),
             qs]
    for k in (1, 3, 5, 10, 20):
        parts.append(topk_mean(R, k)[:, None].astype(np.float32))
    out = np.concatenate(parts, axis=1)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def make_head(params, seed):
    """Instantiate one residual-space classifier from a grid entry."""
    kind = params.get("kind", "gbm")
    if kind == "logreg":
        return Pipeline([
            ("sc", StandardScaler()),
            ("lr", LogisticRegression(C=float(params.get("C", 1.0)),
                                      max_iter=4000, solver="lbfgs")),
        ])
    return HistGradientBoostingClassifier(
        learning_rate=float(params.get("lr", 0.06)),
        max_leaf_nodes=int(params.get("leaves", 31)),
        max_iter=int(params.get("max_iter", 300)),
        l2_regularization=float(params.get("l2", 1.0)),
        early_stopping=False, random_state=int(seed))


def balanced_weights(y):
    """Inverse-frequency sample weights. Anomalous windows are the minority by
    a wide margin; without reweighting the classifier minimises loss by
    predicting 'normal' everywhere."""
    y = np.asarray(y).astype(int)
    w = np.ones(y.size, dtype=np.float64)
    for c in (0, 1):
        n_c = int((y == c).sum())
        if n_c:
            w[y == c] = y.size / (2.0 * n_c)
    return w


def fit_supervised_head(Xf, yf, Xv, yv, seed):
    """Fit every grid entry on the TRAIN-split residual descriptors and keep
    the one with the best validation PR-AUC. Returns (model, params, val_ap).
    Returns (None, None, nan) when either class is missing, in which case the
    caller falls back to the unsupervised scorer alone."""
    yf = np.asarray(yf).astype(int)
    if Xf.shape[0] == 0 or len(np.unique(yf)) < 2 or len(np.unique(np.asarray(yv))) < 2:
        return None, None, float("nan")
    sw = balanced_weights(yf)
    best = (None, None, -np.inf)
    for params in CFG["HEAD_GRID"]:
        try:
            clf = make_head(params, seed)
            if isinstance(clf, Pipeline):
                clf.fit(Xf, yf, lr__sample_weight=sw)
            else:
                clf.fit(Xf, yf, sample_weight=sw)
            ap = safe_ap(yv, clf.predict_proba(Xv)[:, 1])
        except Exception as exc:                # a grid entry must not kill a run
            print(f"[WARN] residual classifier {params} failed: {exc}")
            continue
        if np.isfinite(ap) and ap > best[2]:
            best = (clf, params, float(ap))
    if best[0] is None:
        return None, None, float("nan")
    return best


def logit(p, eps=1e-6):
    """Probabilities are compressed near 0 and 1, which destroys the ranking
    resolution the AUC metrics depend on. The log-odds are not."""
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))
