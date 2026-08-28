# The CPIoT-XAD2025 evaluation protocol

This document states, in full, what is fitted on what. It exists because the
difference between a detector that reports 0.99 and one that *is* 0.99 is
almost never the architecture — it is whether the evaluation quietly let the
model, the normalisation or the threshold see the test data.

## 1. Method summary

Forecast-based anomaly detection on a row-aligned fusion of network-telemetry
features and physical-process sensor/actuator features. A dual-stream model
(1-D CNN over the network stream, stacked LSTM over the sensor stream) is
trained on NORMAL data only to predict the next timestep of both streams.
Anomaly evidence is the robust-normalised one-step forecast residual.

Forecasting rather than reconstruction is deliberate: reconstruction leaves an
identity shortcut, so a low residual can mean "the model copied the input"
rather than "the model understood the dynamics". With `TARGET_MODE="forecast"`
the target row lies outside the input window and no shortcut exists.

Three scorers are built on that shared residual evidence:

- **unsup** — purely unsupervised. Per-stream top-k residual aggregation,
  standardised and combined with a fusion weight α. Uses no anomaly labels at
  any point except to select α / top-k / smoothing on validation.
- **semisup** — a small classifier fitted in residual space on TRAIN-SPLIT
  windows only: held-out normal windows as negatives, the train split's own
  anomalous windows as positives. The forecasting network itself never sees an
  anomaly. This uses supervision the corpus already provides and that the
  unsupervised scorer throws away.
- **hybrid** — a weighted combination of the two, with the mixing weight chosen
  on validation.

## 2. Guarantees

### 2.1 No leakage across splits

Anomaly episodes closer than `MIN_EPISODE_GAP` rows are merged into one
physical attack burst; each burst plus `EPISODE_MARGIN` rows of surrounding
context is an ATOMIC segment that goes to exactly one split. Windows are then
built INSIDE each split, so no window ever spans a split boundary and no attack
is ever partly in train and partly in test.

Segment margins are clipped at the midpoint toward the neighbouring burst, so
atomic segments never overlap.

*Checked by:* `test_no_anomaly_burst_is_split_across_partitions`,
`test_windows_never_cross_a_split_boundary`, `test_segments_never_overlap`.

### 2.2 Anomaly-mass-stratified allocation

Segments are allocated so each split receives a controlled share of the total
anomalous ROW MASS (`ANOM_RATIOS`), not merely a share of the segment count.
Allocation is greedy on mass: segments are visited largest-first and each goes
to whichever split is furthest below its target. A round-robin scheme would let
a single large burst hand one split most of the anomalies and starve another.

Because the detector trains on normal data only, the anomaly mass matters most
in validation and test, where it is actually used — but the train split's
anomalous windows are not wasted either: they are the positive class of the
residual-space classifier.

With fewer than three separated bursts a balanced allocation is impossible; the
pipeline falls back to the classical train-on-normal / test-on-attack protocol,
which is still leakage-free but statistically weak, and says so loudly in the
log.

*Checked by:* `test_validation_and_test_always_receive_anomalies`.

### 2.3 Nothing is fitted on evaluation data

- Median imputation: fitted on TRAIN rows only.
- The z-score scaler: fitted on TRAIN NORMAL rows only.
- The forecasting network: fitted on TRAIN NORMAL windows only, minus the
  calibration holdout.

The calibration holdout of train-normal windows — windows the network never
sees during fitting — is split in two:

- **CALFIT** supplies the residual median/MAD, the score standardisation
  constants, and the negative class of the residual-space classifier.
- **CALTHR** is reserved and, pooled with validation normals, sets every
  decision threshold.

No quantity used to place a threshold is ever estimated on data any model was
fitted to. This is what makes an operating point chosen on validation transfer
to test rather than collapse.

Holdout is by contiguous *run*, not by individual window: windows overlap, so
holding out individual windows would leave the calibration set sharing rows
with the fitting set. Holding out runs leaves only the two windows at each run
boundary in contact.

*Checked by:* `test_calibration_holdout_is_disjoint_and_sized`.

### 2.4 Model selection uses validation only

The fusion weight α, the top-k aggregation width, the temporal smoothing
length, the residual-classifier hyperparameters, the hybrid mixing weight and
the hysteresis ratio are all chosen by validation performance. Test data is
touched once, for reporting.

The full selection grid is exported to `selection_grid.csv`, and
one-factor-at-a-time sweeps around the selected point are exported to
`sensitivity_*.csv` — **on validation, never on test**, so the sharpness of the
selection is documented without turning the test split into a tuning set.

### 2.5 Supervision, where used, is confined to the train split

The residual-space classifier sees only train-split windows. It is never shown
a validation or test window during fitting, and the forecasting network is
never shown an anomalous window at all. Validation supplies the model-selection
criterion only.

The classifier operates on order statistics of the residual matrix (mean, std,
max, a spread of percentiles, several top-k means) rather than raw per-feature
residuals, so the descriptor has the same meaning regardless of how many
features a stream contributes and cannot memorise feature identity.

### 2.6 Ablation under an identical protocol

`fused`, `net_only` and `sen_only` are trained with the same splits, the same
calibration and the same selection procedure, so differences are attributable
to the streams rather than to incidental differences in tuning. Each scorer
also selects its own smoothing length (and, for the hybrid, its mixing weight),
so no scorer is handicapped by another's choices.

### 2.7 Uncertainty is reported

Wilson 95% intervals for precision and recall; percentile-bootstrap 95%
intervals for ROC-AUC and PR-AUC; mean, standard deviation, minimum, maximum
and median over `SEEDS`. The single best run per configuration is identified
explicitly in `best_runs.csv`, so a headline figure can be quoted without
misrepresenting it as the average.

The seed-to-seed spread is a genuine measure of protocol variance, not only of
training stochasticity: `SPLIT_JITTER` perturbs the segment visit order so each
seed produces a different partition, while the mass targets keep every
partition balanced.

### 2.8 Constant-memory windowing

Windows are never materialised. Only window START INDICES are stored, and a
Keras `Sequence` slices each batch out of a single scaled 2-D array via a
zero-copy `sliding_window_view`, so peak RSS is a few GB regardless of corpus
size.

### 2.9 Resumable

Each finished run is checkpointed to `RESULT_DIR/partial/`, so the full
mode × seed protocol can be completed across several sessions. Re-running the
same command reloads completed runs and continues from the first unfinished
one.

## 3. Design choices worth stating

**Window labels follow the predicted row.** Under `LABEL_MODE="target"` a
window takes the label of the row it predicts, because the residual measures
error on that row. Under `"any"` a 30-row window containing a single anomalous
row would be called anomalous even though the predicted row is perfectly
normal, injecting label noise proportional to the window length and suppressing
both precision and recall for reasons that have nothing to do with the
detector.

**Early stopping is monitored on normal validation windows only.** Anomalous
windows are by construction unpredictable, so including them makes the
validation curve flatten on the irreducible attack error instead of reporting
generalisation on normal behaviour.

**Streams are standardised before fusion.** Each stream is aggregated with
top-k and then standardised against its own calibration-normal distribution
*before* α combines them. Otherwise α would be confounded with the arbitrary
difference in residual scale between the two streams, and would not be
interpretable as relative trust.

**Scores are smoothed causally and within runs.** Attacks on a physical plant
persist for many consecutive samples while forecast noise does not, so a
trailing moving average raises the signal-to-noise ratio. The average uses past
and present only, so it stays deployable online, and it never averages across a
discontinuity in the window sequence.

**Hysteresis, because attacks are not independent events.** A single threshold
treats each window as an independent decision. In reality the evidence is
strong at onset, weaker while the process settles into the manipulated state,
and strong again at recovery — so a single threshold fragments one attack into
several short detections and misses its quiet middle. Holding the alarm until
the score falls below a looser exit threshold costs almost no extra false
positives, because on normal data the score rarely crosses the entry threshold
at all.

**A single false-alarm budget is not the honest object.** `TARGET_FPR` is one
point on a curve. `operating_points.csv` re-thresholds the same selected scorer
across `FPR_SWEEP` so the precision/recall trade-off can be read off directly
rather than inferred from one arbitrary choice.

## 4. Failure modes the pipeline announces rather than hides

- Fewer than three separated anomaly bursts → fallback to normal-train /
  anomaly-test, logged as statistically weak.
- Empty calibration holdout → calibration falls back to the fitting windows,
  logged as producing an optimistic false-positive rate.
- Calibration holdout too small to subdivide → reference statistics and
  threshold placement share windows, logged as optimistic.
- No anomalous validation windows → α / top-k / smoothing cannot be selected;
  configured defaults are used and must be reported as un-tuned.
- Fewer than `MIN_TRAIN_ANOM` anomalous train windows → the supervised scorers
  are skipped rather than fitted on a handful of positives.
- A single-class test split → the run aborts rather than reporting a
  meaningless metric.
