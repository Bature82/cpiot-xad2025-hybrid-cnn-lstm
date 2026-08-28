"""Central configuration.

Every hyperparameter used anywhere in the pipeline lives in ``CFG`` and is
written verbatim to ``run_config.json`` next to the results.

``CFG`` is a module-level mutable dictionary: the CLI overrides a handful of
entries (notably the data and result paths) at start-up and every other module
reads the same object, so a run is fully described by the config that is
serialised alongside its outputs.
"""

CFG = {
    # ---- Paths -----------------------------------------------------------
    # Overridden by --data-path / --result-dir on the command line. There is
    # no default that points at anyone's private storage: the CLI requires
    # --data-path explicitly.
    "DATA_PATH": None,
    "RESULT_DIR": "results",

    "FORCE_LABEL_COL": None,
    "LABEL_CANDIDATES": ["label", "Label", "attack", "Attack", "is_anomaly",
                         "anomaly", "Anomaly", "y", "target"],
    # Column-name suffixes identifying the two streams of the fused corpus.
    "NET_SUFFIX": "_nbaiot",
    "SEN_SUFFIX": "_swat",

    # ---- Windowing -------------------------------------------------------
    # WINDOW_SIZE is the history length (in rows) the model conditions on.
    # At the corpus sampling rate a window of 30 rows spans a physically
    # meaningful stretch of process behaviour; attacks on cyber-physical
    # plants develop over tens of seconds, so a very short window cannot
    # separate an attack from ordinary process noise.
    # STRIDE = WINDOW_SIZE // 2 keeps every row covered by at least two
    # windows while halving the number of gradient steps per epoch.
    "WINDOW_SIZE": 30,
    "STRIDE": 15,
    # "forecast": input = rows [s, s+W), target = row s+W (one-step-ahead).
    # "reconstruct": target = row s+W-1, which is inside the input window.
    # Forecasting is preferred: it removes the identity shortcut, so a low
    # residual genuinely means the model understood the dynamics.
    "TARGET_MODE": "forecast",
    # How a window inherits a label from the rows it covers.
    #   "target": the window takes the label of the row it predicts.
    #   "any":    the window is anomalous if ANY covered row is anomalous.
    # "target" is the correct choice for forecast-based detection: the residual
    # measures the error on the target row, so that is the row the label must
    # refer to. Under "any" a 30-row window containing a single anomalous row
    # is called anomalous even though the predicted row is perfectly normal,
    # which injects label noise proportional to the window length and
    # suppresses both precision and recall for reasons that have nothing to do
    # with the detector.
    "LABEL_MODE": "target",

    # ---- Leakage-safe episode-aware split --------------------------------
    "N_BLOCKS": 30,            # granularity for chopping the normal-only remainder
    "EPISODE_MARGIN": 300,     # normal rows of context kept around each anomaly burst
    "MIN_EPISODE_GAP": 50,     # episodes closer than this are one physical burst
    "SPLIT_RATIOS": {"train": 0.80, "val": 0.10, "test": 0.10},
    # Multiplicative jitter applied to the segment visit order during
    # allocation. Zero would make the partition identical for every seed, so
    # the seed-to-seed spread would only measure training stochasticity; a
    # small jitter makes each seed a genuinely different partition while the
    # mass targets below keep every partition balanced.
    "SPLIT_JITTER": 0.35,
    # Share of the total anomalous ROW MASS each split should receive.
    # The forecasting network still trains on normal windows only, but the
    # train split's anomalous windows are no longer wasted: they are the
    # positive class of the residual-space classifier. Train therefore gets a
    # larger share than in a purely unsupervised protocol, while validation and
    # test keep enough mass to estimate metrics with usable intervals.
    "ANOM_RATIOS": {"train": 0.34, "val": 0.33, "test": 0.33},

    # ---- Calibration holdout ---------------------------------------------
    # Fraction of TRAIN NORMAL windows withheld from fitting and used only
    # to estimate residual statistics, score normalisation and the decision
    # threshold. Residuals on data the model has fitted are optimistically
    # small; calibrating on unseen normals is what makes the validation
    # operating point survive the move to test.
    "CALIB_FRAC": 0.20,
    # The calibration holdout is itself split in two. CALIB_THR_FRAC is the
    # share reserved for threshold placement; the remainder supplies the
    # residual median/MAD and the negative class of the residual classifier.
    # Keeping the threshold set disjoint from everything that was fitted is
    # what stops a high-capacity classifier from reporting an optimistic
    # false-positive rate.
    "CALIB_THR_FRAC": 0.50,

    # ---- Training --------------------------------------------------------
    "BATCH_SIZE": 256,
    "EPOCHS": 60,
    "PATIENCE_ES": 8,
    "PATIENCE_RLROP": 4,
    "LR": 3e-4,
    "CLIPNORM": 1.0,
    "HUBER_DELTA": 1.0,
    "DROPOUT": 0.25,
    # Early stopping is monitored on NORMAL validation windows only. Anomalous
    # windows are by construction unpredictable, so including them in the
    # monitored loss makes the validation curve flatten on the irreducible
    # attack error instead of reporting generalisation on normal behaviour.
    "ES_ON_NORMAL_VAL": True,

    # ---- Scoring ---------------------------------------------------------
    "EPS": 1e-9,
    "TARGET_FPR": 0.01,
    # Additional false-positive budgets at which the same selected scorer is
    # re-thresholded and re-evaluated, exported to operating_points.csv. A
    # single arbitrary budget makes a detector look worse than it is; the
    # trade-off curve is the honest object.
    "FPR_SWEEP": [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10],

    # ---- Validation-selected scoring hyperparameters ----------------------
    # alpha (stream fusion weight), top-k (how many feature residuals are
    # aggregated) and the temporal smoothing length are SELECTED on the
    # validation split by PR-AUC rather than fixed a priori. The full grid is
    # exported so the sensitivity of the result to each choice is visible.
    "ALPHA_GRID": [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0],
    "TOPK_GRID": [1, 3, 5, 10, 20],
    # Attacks on a physical plant persist for many consecutive samples, so the
    # useful smoothing lengths are long. The grid reaches 40 windows, which at
    # STRIDE=15 spans roughly 600 rows of process history.
    "SMOOTH_GRID": [1, 2, 3, 5, 8, 12, 20, 30, 40],
    "SELECT_BY": "PR-AUC",       # "PR-AUC" | "ROC-AUC" | "F1"

    # ---- Hysteresis (two-threshold) alarm --------------------------------
    # The alarm is raised when the score crosses the entry threshold and is
    # held until it falls below an exit threshold placed at a looser
    # false-positive budget, TARGET_FPR * multiplier. multiplier = 1 reduces to
    # ordinary single-threshold detection.
    "HYST_GRID": [1, 2, 4, 8, 16, 32],

    # ---- Residual-space classifier (the "semisup" scorer) ----------------
    # Fitted on train-split windows only. The grid is small on purpose: it is
    # selected on validation, and a large grid selected on validation would
    # start to overfit the validation split.
    "SCORERS": ["unsup", "semisup", "hybrid"],
    "HEAD_GRID": [
        {"kind": "gbm",    "lr": 0.06, "leaves": 31, "max_iter": 300, "l2": 1.0},
        {"kind": "gbm",    "lr": 0.10, "leaves": 15, "max_iter": 200, "l2": 5.0},
        {"kind": "gbm",    "lr": 0.03, "leaves": 63, "max_iter": 400, "l2": 1.0},
        {"kind": "logreg", "C": 1.0},
        {"kind": "logreg", "C": 0.05},
    ],
    # Mixing weight of the residual classifier in the hybrid score; 0 is the
    # unsupervised score alone, 1 is the classifier alone.
    "BETA_GRID": [0.0, 0.25, 0.5, 0.75, 0.9, 1.0],
    # Minimum number of anomalous TRAIN windows required before the supervised
    # scorers are attempted. Below this the classifier would be fitted on a
    # handful of positives and its validation score would be noise.
    "MIN_TRAIN_ANOM": 50,

    # ---- Protocol repetition --------------------------------------------
    "SEEDS": [42, 7, 123, 2024, 31337],
    "MODES": ["fused", "net_only", "sen_only"],
    "N_BOOT": 1000,
}
