# CPIoT-XAD2025 — Hybrid CNN/LSTM Cross-Domain Anomaly Detection

Reference implementation of a forecast-based anomaly detector for cyber-physical IoT, operating on a row-aligned fusion of **network telemetry** (N-BaIoT-style features) and **physical-process sensor/actuator** signals (SWaT-style features).

A dual-stream model — a 1-D CNN over the network stream and a stacked LSTM over the sensor stream — is trained on **normal data only** to predict the next timestep of both streams. Anomaly evidence is the robust-normalised one-step forecast residual.

The point of this repository is not only the architecture. It is the **evaluation protocol**: the split, calibration and threshold-placement rules are written so that no quantity used to make a decision is ever estimated on data a model was fitted to. See [`docs/PROTOCOL.md`](docs/PROTOCOL.md) for the full statement, and `tests/` for the guarantees that are checked mechanically.

## Three scorers on shared evidence

All three are built on the same residual evidence and reported side by side under an identical protocol:

| Scorer | Supervision | What it does |
|---|---|---|
| `unsup` | none | Per-stream top-k residual aggregation, standardised and combined with a fusion weight α. No anomaly labels at any point except to select α / top-k / smoothing on validation. |
| `semisup` | train split only | A small classifier fitted in *residual space*: held-out normal windows as negatives, the train split's own anomalous windows as positives. The forecasting network still never sees an anomaly. |
| `hybrid` | train split only | `(1 − β)·unsup + β·semisup`, with β chosen on validation. |

Detection is reported at **four operating points** per scorer: a false-positive-rate-constrained threshold, a validation-F1-optimal threshold, a validation-accuracy-optimal threshold, and a two-threshold hysteresis rule that holds an alarm through the quiet middle of a persistent attack.

## Protocol guarantees

1. **No leakage across splits.** Anomaly episodes closer than `MIN_EPISODE_GAP` rows are merged into one physical attack burst; each burst plus `EPISODE_MARGIN` rows of context is an atomic segment assigned to exactly one split. Windows are built *inside* each split, so no window spans a boundary and no attack is partly in train and partly in test.
2. **Anomaly-mass-stratified allocation.** Segments are allocated by share of the total anomalous *row mass*, not merely segment count.
3. **Nothing is fitted on evaluation data.** Imputation and scaling are fitted on train rows only. The calibration holdout of train-normal windows is split in two: one half supplies residual reference statistics and the classifier's negative class, the other places every decision threshold.
4. **Model selection uses validation only.** α, top-k, smoothing length, classifier hyperparameters, the hybrid mixing weight and the hysteresis ratio are all chosen on validation. Test is touched once, for reporting.
5. **Ablation under an identical protocol.** `fused`, `net_only` and `sen_only` share splits, calibration and selection, so differences are attributable to the streams.
6. **Uncertainty is reported.** Wilson 95% intervals for precision/recall, percentile-bootstrap 95% intervals for ROC-AUC/PR-AUC, and mean/std/min/max across seeds. `best_runs.csv` identifies the single best run explicitly, so a headline figure can be quoted without misrepresenting it as the average.
7. **Constant memory.** Windows are never materialised — only start indices are stored and a Keras `Sequence` slices each batch out of one scaled 2-D array.
8. **Resumable.** Every finished run is checkpointed, so the full mode × seed protocol can be completed across several sessions.

## Installation

```bash
git clone https://github.com/Bature82/cpiot-xad2025-hybrid-cnn-lstm.git
cd cpiot-xad2025-hybrid-cnn-lstm
pip install -r requirements.txt          # or: pip install -e .
```

Python ≥ 3.9. A CUDA-capable GPU is used automatically when TensorFlow can see one; the pipeline runs on CPU but the LSTM stream is slow there.

## Input data

The pipeline expects a single CSV in which each row is one aligned timestep, with:

- network-stream feature columns ending in `_nbaiot` (configurable via `--net-suffix`),
- process-stream feature columns ending in `_swat` (`--sen-suffix`),
- a binary label column (auto-detected from common names, or forced with `--label-col`).

Row order **is** time order — the split and windowing logic depends on it.

The corpus itself is not redistributed here. The underlying public sources are the [N-BaIoT dataset](https://archive.ics.uci.edu/dataset/442/detection+of+iot+botnet+attacks+n+baiot) and the [SWaT testbed dataset](https://itrust.sutd.edu.sg/itrust-labs_datasets/) (SWaT requires a request form).

## Usage

```bash
python -m cpiot_xad.cli \
    --data-path data/CPIoT-XAD2025_fused_hybrid_v5.csv \
    --result-dir results
```

or, after `pip install -e .`, simply `cpiot-xad --data-path ... --result-dir ...`.

Useful options:

```
--modes fused net_only sen_only    # ablation modes to run
--seeds 42 7 123 2024 31337        # protocol repetitions
--scorers unsup semisup hybrid     # which scoring rules to report
--target-fpr 0.01                  # false-alarm budget for the primary threshold
--select-by PR-AUC                 # validation criterion for model selection
--epochs 60 --batch-size 256 --window-size 30 --stride 15
--label-col label --net-suffix _nbaiot --sen-suffix _swat
```

Everything else lives in `cpiot_xad/config.py` and is written verbatim to `run_config.json` beside the results.

A quick smoke run: `--modes fused --seeds 42 --epochs 3`.

## Outputs

All reported numbers are regenerated from one command into one set of files in `--result-dir`:

| File | Contents |
|---|---|
| `metrics_all_runs.csv` | one row per (mode, seed, scorer) with every metric and interval |
| `metrics_summary.csv` | mean / std / min / max / median across seeds |
| `best_runs.csv` | the single best run per configuration under several criteria |
| `operating_points.csv`, `operating_points_summary.csv` | the false-alarm-budget trade-off curve |
| `cis.json` | Wilson and bootstrap 95% intervals for the primary run |
| `selection_grid.csv` | the full validation selection grid |
| `sensitivity_{alpha,topk,smooth}.csv` | one-factor-at-a-time sweeps, on validation |
| `run_config.json` | the exact configuration and runtime |
| `loss_*.png`, `roc_*.png`, `pr_*.png`, `cm_*.png`, `scores_*.png`, `opcurve_*.png` | figures for the primary seed |

### Column conventions

- `VAL_*` — validation split at the FPR-constrained threshold.
- `TEST_*` — test split at the FPR-constrained threshold. The conservative operating point: the threshold admits at most `TARGET_FPR` false alarms on normal data and never consults a test label.
- `TESTOP_*` — test split at the validation-F1-optimal threshold.
- `TESTACC_*` — test split at the validation-accuracy-optimal threshold.
- `TESTHY_*` — test split under the two-threshold hysteresis alarm.
- `*_EventRecall` / `*_EventCount` — supplementary episode-level detection view: an attack event counts as detected if at least one window inside it raises an alarm.

## Repository layout

```
cpiot_xad/
├── config.py         # every hyperparameter, serialised with the results
├── data.py           # loading, label detection, audit, low-variance filter
├── splitting.py      # episodes, atomic bursts, anomaly-mass-stratified allocation
├── windowing.py      # window start indices, calibration holdout (no TensorFlow)
├── sequence.py       # the constant-memory Keras batch generator
├── models.py         # dual-stream CNN/LSTM and the two single-stream ablations
├── scoring.py        # residual evidence, smoothing, thresholds, hysteresis
├── residual_head.py  # the residual-space classifier behind semisup/hybrid
├── metrics.py        # window and event metrics, Wilson and bootstrap intervals
├── figures.py        # ROC, PR, confusion matrix, score histogram, trade-off curve
├── protocol.py       # one complete protocol run = f(mode, seed)
└── cli.py            # resumable mode x seed loop and the export stage
docs/PROTOCOL.md      # the protocol written out in full
notebooks/            # Colab-ready end-to-end notebook
tests/                # protocol guarantees checked mechanically
```

## Tests

```bash
pip install pytest
pytest -q
```

The suite checks the properties the evaluation rests on rather than merely that the code runs: that anomaly bursts are atomic and never split across partitions, that no window or its prediction target crosses a split boundary, that the calibration holdout is disjoint from the fitting set, that smoothing is causal and applied only within contiguous runs, that the threshold realises its false-positive budget, and that the hysteresis rule behaves like a Schmitt trigger.

## Citation

A paper describing this method is in preparation. Until it appears, please cite the repository:

```bibtex
@software{Inuwa_CPIoT_XAD2025,
  author  = {Inuwa, Muhammad Muhammad},
  title   = {{CPIoT-XAD2025}: Hybrid {CNN}/{LSTM} Cross-Domain Anomaly Detection
             for Cyber-Physical {IoT}},
  year    = {2026},
  url     = {https://github.com/Bature82/cpiot-xad2025-hybrid-cnn-lstm},
  version = {1.0.0}
}
```

<!-- TODO: replace the software entry above with the article entry once published. -->

## License

Released under the [MIT License](LICENSE).

## Contact

Muhammad Muhammad Inuwa — bature04764@gmail.com
