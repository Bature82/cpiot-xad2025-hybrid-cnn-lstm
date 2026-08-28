#!/usr/bin/env python3
"""
Alignment diagnostics for CPIoT-XAD2025.

Reads a fused corpus CSV and reports the structural properties the paper
claims for it: stream feature counts, label prevalence, timestamp regularity,
anomaly-episode structure, and whether the corpus supports the episode-aware
split configuration used in the reported runs.

This script does NOT build the corpus. It checks one that has already been
built, so that a reader who reconstructs CPIoT-XAD2025 from N-BaIoT and SWaT
can confirm the result matches what was evaluated.

Usage
-----
    python fusion/verify_alignment.py --data path/to/CPIoT-XAD2025_fused.csv
    python fusion/verify_alignment.py --data fused.csv --json diagnostics.json

Exit status is 0 when every check passes, 1 when any FAIL is reported.
"""

import argparse
import json
import sys

import numpy as np
import pandas as pd

# Conventions fixed by the reference pipeline (see config.py / PROTOCOL.md).
NET_SUFFIX = "_nbaiot"
SEN_SUFFIX = "_swat"
LABEL_CANDIDATES = ["label", "Label", "attack", "Attack",
                    "is_anomaly", "anomaly", "Anomaly", "y", "target"]
TIME_CANDIDATES = ["timestamp", "Timestamp", "time", "Time", "datetime", "ts"]

# Episode-aware split parameters used in the reported runs.
MIN_EPISODE_GAP = 50      # runs closer than this merge into one burst
EPISODE_MARGIN = 300      # normal context padded around each burst
N_BLOCKS = 30             # granularity for the normal-only remainder
LOW_VAR_THRESHOLD = 1e-12

_results = []


def check(name, ok, detail, hard=True):
    """Record one diagnostic. hard=False downgrades a failure to a warning."""
    status = "PASS" if ok else ("FAIL" if hard else "WARN")
    _results.append({"check": name, "status": status, "detail": detail})
    print(f"  [{status}] {name}: {detail}")
    return ok


def find_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def find_episodes(y):
    """Half-open [start, end) intervals of contiguous runs of label 1."""
    y = np.asarray(y).astype(int)
    d = np.flatnonzero(np.diff(np.r_[0, y, 0]))
    return list(zip(d[0::2], d[1::2]))


def merge_bursts(eps, min_gap):
    if not eps:
        return []
    out = [list(eps[0])]
    for s, e in eps[1:]:
        if s - out[-1][1] < min_gap:
            out[-1][1] = e
        else:
            out.append([s, e])
    return [tuple(x) for x in out]


def padded_blocks(bursts, n_rows, margin):
    """Bursts padded by `margin` rows, clipped at midpoints so none overlap."""
    blocks = []
    for k, (s, e) in enumerate(bursts):
        lo, hi = max(0, s - margin), min(n_rows, e + margin)
        if k > 0:
            lo = max(lo, (bursts[k - 1][1] + s) // 2)
        if k < len(bursts) - 1:
            hi = min(hi, (e + bursts[k + 1][0]) // 2)
        blocks.append((lo, hi))
    return blocks


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="path to the fused corpus CSV")
    ap.add_argument("--net-suffix", default=NET_SUFFIX)
    ap.add_argument("--sen-suffix", default=SEN_SUFFIX)
    ap.add_argument("--label-col", default=None,
                    help="force the label column instead of auto-detecting")
    ap.add_argument("--json", default=None, help="also write results to this JSON file")
    args = ap.parse_args()

    print(f"\nCPIoT-XAD2025 alignment diagnostics\nfile: {args.data}\n")
    df = pd.read_csv(args.data)
    n_rows = len(df)
    print(f"  rows: {n_rows}    columns: {df.shape[1]}\n")

    # ---- 1. stream columns -------------------------------------------------
    net = [c for c in df.columns
           if str(c).lower().endswith(args.net_suffix)
           and pd.api.types.is_numeric_dtype(df[c])]
    sen = [c for c in df.columns
           if str(c).lower().endswith(args.sen_suffix)
           and pd.api.types.is_numeric_dtype(df[c])]
    check("network columns present", len(net) > 0,
          f"{len(net)} numeric columns ending in '{args.net_suffix}'")
    check("sensor columns present", len(sen) > 0,
          f"{len(sen)} numeric columns ending in '{args.sen_suffix}'")

    # Counts after the low-variance filter the pipeline applies.
    net_v = [c for c in net if df[c].var(numeric_only=True) > LOW_VAR_THRESHOLD]
    sen_v = [c for c in sen if df[c].var(numeric_only=True) > LOW_VAR_THRESHOLD]
    check("network features after low-variance filter", len(net_v) > 0,
          f"{len(net_v)} of {len(net)} retained "
          f"({len(net) - len(net_v)} constant)")
    check("sensor features after low-variance filter", len(sen_v) > 0,
          f"{len(sen_v)} of {len(sen)} retained "
          f"({len(sen) - len(sen_v)} constant)")

    # ---- 2. label column ---------------------------------------------------
    label_col = args.label_col or find_column(df, LABEL_CANDIDATES)
    if not check("label column found", label_col is not None,
                 f"'{label_col}'" if label_col else
                 f"none of {LABEL_CANDIDATES} present"):
        finish(args)
        return

    y = pd.to_numeric(df[label_col], errors="coerce").fillna(0)
    y = (y > 0).astype(int).to_numpy()
    n_anom = int(y.sum())
    prevalence = n_anom / max(1, n_rows)
    check("label is binary and non-degenerate", 0 < n_anom < n_rows,
          f"{n_anom} anomalous rows of {n_rows} "
          f"(row prevalence {prevalence:.4f})")

    # ---- 3. timestamps -----------------------------------------------------
    time_col = find_column(df, TIME_CANDIDATES)
    if time_col is None:
        check("timestamp column", False,
              f"none of {TIME_CANDIDATES} present; alignment cannot be "
              "verified from the file alone", hard=False)
    else:
        ts = pd.to_datetime(df[time_col], errors="coerce")
        n_bad = int(ts.isna().sum())
        check("timestamps parse", n_bad == 0,
              f"'{time_col}': {n_bad} unparsable of {n_rows}")
        if n_bad == 0:
            check("timestamps strictly increasing",
                  bool((ts.diff().dropna() > pd.Timedelta(0)).all()),
                  "monotonic" if bool((ts.diff().dropna() > pd.Timedelta(0)).all())
                  else "non-monotonic rows present")
            step = ts.diff().dropna().dt.total_seconds()
            if len(step):
                check("sampling interval regular",
                      float(step.std()) < 1e-6,
                      f"median {step.median():.3f}s, "
                      f"sd {step.std():.6f}s "
                      f"(min {step.min():.3f}, max {step.max():.3f})",
                      hard=False)

    # ---- 4. missing values -------------------------------------------------
    feat = net + sen
    n_nan = int(df[feat].isna().sum().sum())
    check("no missing feature values", n_nan == 0,
          f"{n_nan} NaN cells across {len(feat)} feature columns",
          hard=False)

    # ---- 5. episode structure ---------------------------------------------
    eps = find_episodes(y)
    bursts = merge_bursts(eps, MIN_EPISODE_GAP)
    blocks = padded_blocks(bursts, n_rows, EPISODE_MARGIN)
    check("anomaly episodes present", len(eps) > 0,
          f"{len(eps)} contiguous anomalous runs")
    check("bursts after merging gaps < %d rows" % MIN_EPISODE_GAP,
          len(bursts) >= 3,
          f"{len(bursts)} separated bursts "
          "(3 or more are needed for a stratified three-way split)")
    if bursts:
        lens = np.array([e - s for s, e in bursts])
        print(f"       burst lengths: min {lens.min()}, median "
              f"{int(np.median(lens))}, max {lens.max()} rows")
        mass = np.array([int(y[s:e].sum()) for s, e in blocks])
        print(f"       anomaly mass per padded block: min {mass.min()}, "
              f"max {mass.max()}, total {mass.sum()}")

    covered = sum(hi - lo for lo, hi in blocks)
    check("padded blocks leave a normal-only remainder",
          covered < n_rows,
          f"{covered} of {n_rows} rows inside padded blocks "
          f"({100.0 * covered / max(1, n_rows):.1f}%)")
    blk = max(50, n_rows // max(N_BLOCKS, 1))
    check("normal remainder supports the block granularity",
          (n_rows - covered) >= blk,
          f"remainder {n_rows - covered} rows, block size {blk} rows")

    finish(args)


def finish(args):
    n_fail = sum(1 for r in _results if r["status"] == "FAIL")
    n_warn = sum(1 for r in _results if r["status"] == "WARN")
    print(f"\n  {len(_results)} checks: "
          f"{len(_results) - n_fail - n_warn} passed, "
          f"{n_warn} warnings, {n_fail} failures\n")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(_results, f, indent=2)
        print(f"  written: {args.json}\n")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
