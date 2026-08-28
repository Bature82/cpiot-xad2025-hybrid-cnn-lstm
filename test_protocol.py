"""Unit tests for the protocol guarantees that do not require TensorFlow.

These check the properties the evaluation depends on: that no window crosses a
split boundary, that anomaly bursts are atomic, that smoothing is causal and
run-local, and that the hysteresis rule behaves like a Schmitt trigger.

Run with:  pytest -q
"""

import numpy as np
import pytest

from cpiot_xad.config import CFG
from cpiot_xad.metrics import compute_metrics, event_metrics, wilson_ci
from cpiot_xad.scoring import (
    hysteresis_predict, smooth_scores, threshold_from_normals, topk_mean,
)
from cpiot_xad.splitting import assign_blocks, build_atomic_segments, find_episodes
from cpiot_xad.windowing import (
    contiguous_runs, holdout_calibration, target_offset,
    window_starts_within_split,
)


def synthetic_labels(n=20000, n_bursts=6, burst_len=200, rng_seed=0):
    """A timeline with `n_bursts` well-separated anomaly bursts."""
    y = np.zeros(n, dtype=int)
    step = n // (n_bursts + 1)
    for b in range(n_bursts):
        s = step * (b + 1)
        y[s:s + burst_len] = 1
    return y


# --------------------------------------------------------------------------
# Episode and segment construction
# --------------------------------------------------------------------------
def test_find_episodes_recovers_runs():
    y = np.array([0, 1, 1, 0, 0, 1, 0])
    assert find_episodes(y) == [(1, 3), (5, 6)]


def test_find_episodes_handles_edges():
    assert find_episodes(np.array([1, 1, 0])) == [(0, 2)]
    assert find_episodes(np.array([0, 1, 1])) == [(1, 3)]
    assert find_episodes(np.zeros(5, dtype=int)) == []


def test_nearby_episodes_merge_into_one_burst():
    y = np.zeros(3000, dtype=int)
    y[1000:1050] = 1
    y[1060:1100] = 1          # gap of 10 < MIN_EPISODE_GAP
    segs, n_eps = build_atomic_segments(y)
    assert n_eps == 2
    assert len(segs) == 1     # merged into a single physical burst


def test_segments_never_overlap():
    y = synthetic_labels()
    segs, _ = build_atomic_segments(y)
    for (lo1, hi1), (lo2, hi2) in zip(segs, segs[1:]):
        assert hi1 <= lo2


def test_every_anomalous_row_is_inside_a_segment():
    y = synthetic_labels()
    segs, _ = build_atomic_segments(y)
    covered = np.zeros(len(y), dtype=bool)
    for lo, hi in segs:
        covered[lo:hi] = True
    assert covered[y == 1].all()


# --------------------------------------------------------------------------
# Split allocation
# --------------------------------------------------------------------------
def test_split_covers_every_row_exactly_once():
    y = synthetic_labels()
    split, _, _ = assign_blocks(y, CFG["N_BLOCKS"], CFG["SPLIT_RATIOS"], seed=42)
    assert len(split) == len(y)
    assert set(np.unique(split)) <= {"train", "val", "test"}
    assert not any(v is None for v in split)


def test_no_anomaly_burst_is_split_across_partitions():
    """The central leakage guarantee: an attack burst plus its context margin
    belongs to exactly one split."""
    y = synthetic_labels()
    split, segs, _ = assign_blocks(y, CFG["N_BLOCKS"], CFG["SPLIT_RATIOS"], seed=42)
    for lo, hi in segs:
        assert len(set(split[lo:hi])) == 1


@pytest.mark.parametrize("seed", [42, 7, 123, 2024, 31337])
def test_validation_and_test_always_receive_anomalies(seed):
    y = synthetic_labels()
    split, _, _ = assign_blocks(y, CFG["N_BLOCKS"], CFG["SPLIT_RATIOS"], seed)
    for part in ("val", "test"):
        assert y[split == part].sum() > 0


# --------------------------------------------------------------------------
# Windowing
# --------------------------------------------------------------------------
def test_windows_never_cross_a_split_boundary():
    y = synthetic_labels()
    split, _, _ = assign_blocks(y, CFG["N_BLOCKS"], CFG["SPLIT_RATIOS"], seed=42)
    W, S = CFG["WINDOW_SIZE"], CFG["STRIDE"]
    off = target_offset(W)
    for part in ("train", "val", "test"):
        starts, _ = window_starts_within_split(y, split, part, W, S)
        for s in starts:
            # the whole input window AND its prediction target lie in one split
            assert len(set(split[s:s + off + 1])) == 1
            assert split[s] == part


def test_window_label_matches_the_predicted_row():
    y = synthetic_labels()
    split, _, _ = assign_blocks(y, CFG["N_BLOCKS"], CFG["SPLIT_RATIOS"], seed=42)
    W, S = CFG["WINDOW_SIZE"], CFG["STRIDE"]
    off = target_offset(W)
    starts, labels = window_starts_within_split(y, split, "test", W, S)
    assert np.array_equal(labels, y[starts + off])


def test_contiguous_runs_split_on_gaps():
    starts = np.array([0, 15, 30, 100, 115])
    runs = contiguous_runs(starts, 15)
    assert [list(r) for r in runs] == [[0, 1, 2], [3, 4]]


def test_calibration_holdout_is_disjoint_and_sized():
    starts = np.arange(0, 15 * 400, 15)
    fit, cal = holdout_calibration(starts, 15, 0.2, seed=0)
    assert set(fit).isdisjoint(set(cal))
    assert len(fit) + len(cal) == len(starts)
    assert 0.1 * len(starts) <= len(cal) <= 0.35 * len(starts)


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
def test_topk_mean_matches_manual_computation():
    x = np.array([[1.0, 5.0, 3.0, 2.0]])
    assert topk_mean(x, 1)[0] == pytest.approx(5.0)
    assert topk_mean(x, 2)[0] == pytest.approx(4.0)
    # k larger than the feature count degrades to the plain mean
    assert topk_mean(x, 99)[0] == pytest.approx(2.75)


def test_smoothing_is_causal_and_run_local():
    starts = np.array([0, 15, 30, 500])       # last window starts a new run
    scores = np.array([1.0, 3.0, 5.0, 100.0])
    out = smooth_scores(starts, scores, 15, k=2)
    assert out[0] == pytest.approx(1.0)       # first element: no past to average
    assert out[1] == pytest.approx(2.0)       # (1+3)/2
    assert out[2] == pytest.approx(4.0)       # (3+5)/2
    assert out[3] == pytest.approx(100.0)     # separate run, unaffected


def test_smoothing_k1_is_identity():
    starts = np.arange(0, 150, 15)
    scores = np.random.default_rng(0).normal(size=starts.size)
    assert np.allclose(smooth_scores(starts, scores, 15, 1), scores)


def test_threshold_respects_the_false_positive_budget():
    rng = np.random.default_rng(0)
    normals = rng.normal(size=100000)
    thr = threshold_from_normals(normals, 0.01)
    realised = float((normals > thr).mean())
    assert realised == pytest.approx(0.01, abs=2e-3)


def test_hysteresis_holds_the_alarm_through_a_dip():
    starts = np.arange(0, 15 * 6, 15)
    # crosses hi at index 1, dips between hi and lo, recovers, then drops below lo
    scores = np.array([0.0, 10.0, 3.0, 9.0, 0.5, 0.0])
    pred = hysteresis_predict(starts, scores, 15, thr_hi=5.0, thr_lo=1.0)
    assert list(pred) == [0, 1, 1, 1, 0, 0]


def test_hysteresis_reduces_to_single_threshold_when_lo_equals_hi():
    starts = np.arange(0, 15 * 4, 15)
    scores = np.array([0.0, 10.0, 3.0, 9.0])
    pred = hysteresis_predict(starts, scores, 15, thr_hi=5.0, thr_lo=5.0)
    assert list(pred) == list((scores > 5.0).astype(int))


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def test_compute_metrics_confusion_counts():
    y = np.array([0, 0, 1, 1])
    p = np.array([0, 1, 1, 0])
    m = compute_metrics(y, p)
    assert (m["TN"], m["FP"], m["FN"], m["TP"]) == (1, 1, 1, 1)
    assert m["Precision"] == pytest.approx(0.5)
    assert m["Recall"] == pytest.approx(0.5)
    assert m["FPR"] == pytest.approx(0.5)


def test_event_recall_counts_an_episode_once():
    starts = np.arange(0, 15 * 8, 15)
    y = np.array([0, 1, 1, 1, 0, 1, 1, 0])
    pred = np.array([0, 0, 1, 0, 0, 0, 0, 0])     # one hit inside episode 1 only
    ev = event_metrics(starts, y, pred, 15)
    assert ev["EventCount"] == 2
    assert ev["EventsDetected"] == 1
    assert ev["EventRecall"] == pytest.approx(0.5)


def test_wilson_interval_brackets_the_point_estimate():
    lo, hi = wilson_ci(30, 100)
    assert lo < 0.30 < hi
    assert 0.0 <= lo and hi <= 1.0
    # a wider interval when the sample is small
    lo_s, hi_s = wilson_ci(3, 10)
    assert (hi_s - lo_s) > (hi - lo)
