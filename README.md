# Reported results

Every table and figure in the paper is generated from the files in
`results_v8/`, which is the complete output of one execution of the protocol:
45 runs spanning three modality configurations (`fused`, `net_only`,
`sen_only`), three scoring rules (`unsup`, `semisup`, `hybrid`) and five
episode rotations (seeds 42, 7, 123, 2024, 31337). Total runtime 1,022 s on a
single mid-range GPU.

These files are committed so that the reported numbers can be checked without
access to SWaT and without re-running the pipeline.

| File | Contents |
|---|---|
| `metrics_all_runs.csv` | One row per run (45 × 107 columns): all metrics at all four operating points, selected hyper-parameters, split sizes and confidence intervals. |
| `metrics_summary.csv` | Mean, standard deviation, min, max and median across seeds, grouped by mode and scorer. |
| `best_runs.csv` | The single best run per (mode, scorer) under each of several criteria. |
| `operating_points.csv` | The false-alarm budget sweep: every scorer re-thresholded at each target FPR. |
| `operating_points_summary.csv` | The same, aggregated across seeds. |
| `selection_grid.csv` | The full validation selection grid over α, top-k, smoothing and β. |
| `sensitivity_alpha.csv` | One-factor-at-a-time sweep of the fusion weight α on validation. |
| `sensitivity_topk.csv`, `sensitivity_smooth.csv` | The same for top-k and smoothing. |
| `cis.json` | Wilson and bootstrap intervals for the primary rotation. |
| `run_config.json` | Every hyper-parameter used, written verbatim by the run. |
| `history_*_seed42.csv` | Per-epoch training and validation loss for the primary rotation. |
| `*.png` | Loss, ROC, PR, confusion, score-distribution and operating-point figures. |

## Column conventions

| Prefix | Operating point |
|---|---|
| `VAL_` | validation split at the FPR-constrained threshold |
| `TEST_` | test split at the FPR-constrained threshold (the deployment default, ρ = 0.01) |
| `TESTOP_` | test split at the validation-F1-optimal threshold |
| `TESTACC_` | test split at the validation-accuracy-optimal threshold |
| `TESTHY_` | test split under the two-threshold hysteresis alarm |

`*_EventRecall` and `*_EventCount` are the supplementary episode-level view.

## Reproducing the paper's tables

```python
import pandas as pd
m = pd.read_csv("results/results_v8/metrics_all_runs.csv")
u = m[m.scorer == "unsup"]
print(u.groupby("mode")[["TEST_ROC-AUC", "TEST_PR-AUC"]].agg(["mean", "std"]))
```
