"""Window index construction and the calibration holdout.

Windows are never materialised. Only window START INDICES are stored; the Keras
Sequence in :mod:`cpiot_xad.sequence` slices each batch out of a single scaled
2-D array, so peak RSS is a few GB regardless of corpus size.

This module is deliberately free of any TensorFlow import so that the index
logic can be tested without a deep-learning runtime installed.
"""

import numpy as np

from .config import CFG


def target_offset(w):
    """Row offset of the prediction target relative to the window start."""
    return w if CFG["TARGET_MODE"] == "forecast" else (w - 1)


def window_starts_within_split(y, split_per_row, split_name, w, stride):
    """Return window START indices (into the full row timeline) and window
    labels, taken only inside maximal contiguous runs belonging to
    split_name, so no window and no prediction target ever crosses a split
    boundary.

    Labels follow CFG["LABEL_MODE"]: "target" gives the window the label of the
    row it predicts, "any" marks the window anomalous if any covered row is
    anomalous."""
    off = target_offset(w)
    span = off + 1                       # rows covered by one training example
    label_mode = CFG.get("LABEL_MODE", "target")
    mask = (split_per_row == split_name)
    starts_all, labels_all = [], []
    i, N = 0, len(mask)
    while i < N:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < N and mask[j]:
            j += 1
        seg_len = j - i
        if seg_len >= span:
            seg_y = np.asarray(y[i:j])
            n_pos = seg_len - span + 1            # valid starts inside the segment
            idx = np.arange(0, n_pos, stride)
            if label_mode == "target":
                lab = seg_y[idx + off]
            else:
                win_max = np.lib.stride_tricks.sliding_window_view(
                    seg_y, span).max(axis=1)
                lab = win_max[idx]
            starts_all.append(i + idx)
            labels_all.append(lab)
        i = j
    if not starts_all:
        return np.array([], dtype=np.int64), np.array([], dtype=int)
    return (np.concatenate(starts_all).astype(np.int64),
            np.concatenate(labels_all).astype(int))


def contiguous_runs(sorted_starts, stride):
    """Index groups of windows that are consecutive in time (start indices
    exactly `stride` apart). Used both to carve a calibration holdout that
    does not overlap the fitting windows and to smooth scores only along
    genuinely adjacent windows."""
    s = np.asarray(sorted_starts)
    if s.size == 0:
        return []
    brk = np.flatnonzero(np.diff(s) != stride) + 1
    return np.split(np.arange(s.size), brk)


def holdout_calibration(starts, stride, frac, seed):
    """Withhold whole contiguous runs of train-normal windows for calibration.
    Holding out runs rather than individual windows keeps the calibration set
    almost disjoint from the fitting set despite window overlap: only the two
    windows at each run boundary share rows with the other set."""
    s = np.sort(np.asarray(starts, dtype=np.int64))
    if s.size == 0 or frac <= 0:
        return s, np.array([], dtype=np.int64)
    groups = contiguous_runs(s, stride)
    # subdivide long runs so at least ~20 pieces exist to choose from
    max_len = max(1, s.size // 20)
    pieces = []
    for g in groups:
        for a in range(0, len(g), max_len):
            pieces.append(g[a:a + max_len])
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(pieces))
    need, got, chosen = int(round(frac * s.size)), 0, []
    for pi in order:
        if got >= need:
            break
        chosen.append(pieces[pi])
        got += len(pieces[pi])
    cal_mask = np.zeros(s.size, dtype=bool)
    if chosen:
        cal_mask[np.concatenate(chosen)] = True
    if cal_mask.all():                      # degenerate corpus guard
        cal_mask[: s.size // 2] = False
    return s[~cal_mask], s[cal_mask]
