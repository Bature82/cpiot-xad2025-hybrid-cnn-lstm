"""One complete protocol run = f(mode, seed).

Splitting, imputation, scaling, calibration, selection and evaluation are all
repeated inside :func:`run_protocol`, so every statistic is re-fitted on the
current seed's train split alone.
"""

import gc
import json
import os
import warnings

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
)

from .config import CFG
from .figures import save_detection_figures, save_learning_curve
from .metrics import (
    bootstrap_ci, compute_metrics, event_metrics, safe_ap, safe_auc, wilson_ci,
)
from .models import MemCallback, build_model
from .residual_head import (
    fit_supervised_head, logit, residual_features,
)
from .scoring import (
    apply_mad, best_threshold, fit_mad, fit_residual_stats, hysteresis_predict,
    residual_matrix, smooth_scores, threshold_from_normals, topk_mean,
)
from .sequence import WindowSeq
from .splitting import assign_blocks
from .utils import mem
from .windowing import (
    holdout_calibration, target_offset, window_starts_within_split,
)


def run_protocol(mode, seed, X_np, y_rows, n_net, n_sen, make_figures=False):
    """Run the full protocol for one (mode, seed) pair.

    Returns
    -------
    rows, grid_rows, op_rows, sens_alpha, sens_topk, sens_smooth
    """
    np.random.seed(seed)
    tf.random.set_seed(seed)
    result_dir = CFG["RESULT_DIR"]

    # ---- split the row timeline BEFORE any windowing --------------------
    split_per_row, segs, seg_role = assign_blocks(
        y_rows, CFG["N_BLOCKS"], CFG["SPLIT_RATIOS"], seed)
    tr_rows = split_per_row == "train"

    # ---- impute + scale, fitted on train rows only ----------------------
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN feature slice
        med_train = np.nanmedian(X_np[tr_rows], axis=0)
    med_train = np.where(np.isfinite(med_train), med_train, 0.0).astype(np.float32)
    data = np.where(np.isnan(X_np), med_train[None, :], X_np)

    tr_norm_rows = tr_rows & (y_rows == 0)
    scaler = StandardScaler().fit(data[tr_norm_rows])
    data -= scaler.mean_.astype(np.float32)      # z-score, in place
    data /= scaler.scale_.astype(np.float32)

    # ---- window start indices inside each split -------------------------
    W, S = CFG["WINDOW_SIZE"], CFG["STRIDE"]
    OFF = target_offset(W)
    tr_starts, ytr = window_starts_within_split(y_rows, split_per_row, "train", W, S)
    va_starts, yva = window_starts_within_split(y_rows, split_per_row, "val", W, S)
    te_starts, yte = window_starts_within_split(y_rows, split_per_row, "test", W, S)

    if len(np.unique(yte)) < 2:
        raise SystemExit(f"[ERROR] test split contains a single class (seed={seed}).")
    val_has_anom = len(np.unique(yva)) >= 2
    if not val_has_anom:
        print("[WARN] validation split has no anomalous windows. The FPR "
              "threshold is still valid because it uses normals only, but "
              "alpha / top-k / smoothing cannot be selected on validation; "
              "the configured defaults are used and must be reported as "
              "un-tuned.")

    # ---- calibration holdout of train normals ---------------------------
    # Two nested holdouts. First the train-normal windows are split into the
    # windows the network is fitted on and a calibration holdout it never sees.
    # Then the calibration holdout is itself split: CALFIT supplies the
    # residual median/MAD, the score standardisation constants and the negative
    # class of the residual classifier; CALTHR is reserved and, pooled with the
    # validation normals, places every decision threshold. Nothing that sets a
    # threshold has ever been fitted.
    tr_norm_starts = tr_starts[ytr == 0]
    fit_starts, cal_starts = holdout_calibration(
        tr_norm_starts, S, CFG["CALIB_FRAC"], seed)
    if cal_starts.size == 0 or fit_starts.size == 0:
        # degenerate corpus: fall back to calibrating on the fitting windows.
        # Legitimate but optimistic - residuals on fitted data are smaller than
        # on unseen data, so the realised false positive rate will exceed the
        # target. Flagged here so it cannot pass unnoticed.
        print("[WARN] calibration holdout is empty; calibrating on the fitting "
              "windows instead. The FPR target will be optimistic.")
        fit_starts = np.sort(tr_norm_starts)
        cal_starts = fit_starts
    calfit_starts, calthr_starts = holdout_calibration(
        cal_starts, S, CFG["CALIB_THR_FRAC"], seed + 1)
    if calfit_starts.size == 0 or calthr_starts.size == 0:
        print("[WARN] calibration holdout too small to subdivide; the same "
              "windows are used for reference statistics and for threshold "
              "placement. Reported false-positive rates will be optimistic.")
        calfit_starts = calthr_starts = np.sort(cal_starts)
    va_norm_starts = va_starts[yva == 0]
    tra_starts = tr_starts[ytr == 1]      # positives for the residual classifier

    split_doc = {
        "seed": seed, "n_blocks": CFG["N_BLOCKS"],
        "window_size": W, "stride": S, "target_mode": CFG["TARGET_MODE"],
        "label_mode": CFG.get("LABEL_MODE", "target"),
        "fit_windows": int(len(fit_starts)), "calib_windows": int(len(cal_starts)),
        "calfit_windows": int(len(calfit_starts)),
        "calthr_windows": int(len(calthr_starts)),
        "train_windows": int(len(ytr)), "train_anom": int(ytr.sum()),
        "val_windows": int(len(yva)), "val_anom": int(yva.sum()),
        "test_windows": int(len(yte)), "test_anom": int(yte.sum()),
    }
    print(f"[SPLIT] {split_doc}")
    mem("after windowing")

    # ---- prediction targets (small: one row per window) ------------------
    T_cf = data[calfit_starts + OFF]
    T_ct = data[calthr_starts + OFF]
    T_ta = data[tra_starts + OFF] if tra_starts.size else data[:0]
    T_va = data[va_starts + OFF]
    T_te = data[te_starts + OFF]

    # ---- train on normal windows only -----------------------------------
    model = build_model(mode, (W, n_net), (W, n_sen),
                        CFG["LR"], CFG["CLIPNORM"], CFG["HUBER_DELTA"],
                        CFG["DROPOUT"])

    cbs = [EarlyStopping(monitor="val_loss", patience=CFG["PATIENCE_ES"],
                         restore_best_weights=True),
           ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                             patience=CFG["PATIENCE_RLROP"]),
           MemCallback(mem)]
    if make_figures:
        cbs.append(ModelCheckpoint(
            os.path.join(result_dir, f"best_{mode}_seed{seed}.keras"),
            save_best_only=True))

    es_starts = va_norm_starts if (CFG["ES_ON_NORMAL_VAL"] and len(va_norm_starts) > 0) \
        else va_starts
    train_seq = WindowSeq(data, fit_starts, W, n_net, mode, CFG["BATCH_SIZE"],
                          shuffle=True, seed=seed)
    val_seq = WindowSeq(data, es_starts, W, n_net, mode, CFG["BATCH_SIZE"])
    hist = model.fit(train_seq, validation_data=val_seq,
                     epochs=CFG["EPOCHS"], callbacks=cbs, verbose=0)
    mem("after fit")

    # ---- predict once per split; the selection grid reuses these ---------
    PRED_BS = max(CFG["BATCH_SIZE"], 1024)

    def _predict(starts):
        seq = WindowSeq(data, starts, W, n_net, mode, PRED_BS, with_targets=False)
        return model.predict(seq, verbose=0)

    SPLITS = ("calfit", "calthr", "tranom", "val", "test")
    START_OF = {"calfit": calfit_starts, "calthr": calthr_starts,
                "tranom": tra_starts, "val": va_starts, "test": te_starts}
    TGT_OF = {"calfit": T_cf, "calthr": T_ct, "tranom": T_ta,
              "val": T_va, "test": T_te}
    PRED = {k: (_predict(v) if v.size else None) for k, v in START_OF.items()}

    # ---- per-stream normalised residual evidence ------------------------
    # Reference median/MAD come from CALFIT, i.e. from normal data the network
    # never fitted, so residual magnitudes on every other split are measured on
    # the same scale as the one the threshold will be set on.
    if mode == "fused":
        stream_slices = {"net": (0, n_net), "sen": (n_net, None)}
    elif mode == "net_only":
        stream_slices = {"net": (0, n_net)}
    else:
        stream_slices = {"sen": (n_net, None)}

    def _pred_for(split, si):
        p = PRED[split]
        if p is None:
            return None
        return p[si] if mode == "fused" else p

    streams = {}                     # name -> {split: residual matrix}
    for si, (name, (lo, hi)) in enumerate(stream_slices.items()):
        p_cf, t_cf = _pred_for("calfit", si), TGT_OF["calfit"][:, lo:hi]
        med, mad = fit_residual_stats(p_cf, t_cf, CFG["EPS"])
        per_split = {}
        for sp in SPLITS:
            p = _pred_for(sp, si)
            if p is None or START_OF[sp].size == 0:
                per_split[sp] = np.zeros((0, t_cf.shape[1]), dtype=np.float32)
            else:
                per_split[sp] = residual_matrix(p, TGT_OF[sp][:, lo:hi], med, mad)
        streams[name] = per_split
    mem("after residual evidence")

    # ---- score assembly -------------------------------------------------
    # Each stream is aggregated with top-k, then standardised against its own
    # calibration-normal distribution BEFORE fusion. Without this the fusion
    # weight alpha would be confounded with the arbitrary difference in scale
    # between network and process residuals, and alpha would not be
    # interpretable as a relative trust in the two streams.
    EVAL = ("calthr", "val", "test")
    _agg_cache = {}

    def aggregated(topk):
        """Per-stream top-k residual score, standardised against the CALFIT
        normal distribution. Standardising before fusion is what makes alpha a
        scale-free statement about how much each stream is trusted rather than
        an accident of the two streams' residual magnitudes."""
        if topk in _agg_cache:
            return _agg_cache[topk]
        z = {}
        for name, per_split in streams.items():
            mu, md = fit_mad(topk_mean(per_split["calfit"], topk))
            z[name] = {sp: apply_mad(topk_mean(per_split[sp], topk), mu, md)
                       for sp in EVAL}
        _agg_cache[topk] = z
        return z

    def unsup_raw(alpha, topk):
        z = aggregated(topk)
        if mode == "fused":
            return {sp: alpha * z["net"][sp] + (1.0 - alpha) * z["sen"][sp]
                    for sp in EVAL}
        key = "net" if mode == "net_only" else "sen"
        return z[key]

    STARTS_EVAL = {"calthr": calthr_starts, "val": va_starts, "test": te_starts}

    def smoothed(raw, k):
        return {sp: smooth_scores(STARTS_EVAL[sp], raw[sp], S, k) for sp in EVAL}

    # ---- select alpha / top-k / smoothing on validation ------------------
    alpha_grid = CFG["ALPHA_GRID"] if mode == "fused" else \
        ([1.0] if mode == "net_only" else [0.0])
    grid_rows, best = [], None
    for topk in CFG["TOPK_GRID"]:
        for alpha in alpha_grid:
            raw = unsup_raw(alpha, topk)
            for sm in CFG["SMOOTH_GRID"]:
                sc = smoothed(raw, sm)
                thr = threshold_from_normals(
                    np.concatenate([sc["calthr"], sc["val"][yva == 0]]),
                    CFG["TARGET_FPR"])
                vm = compute_metrics(yva, (sc["val"] > thr).astype(int), sc["val"])
                crit = vm.get(CFG["SELECT_BY"], np.nan)
                crit = -np.inf if not np.isfinite(crit) else crit
                grid_rows.append({"mode": mode, "seed": seed, "scorer": "unsup",
                                  "alpha": alpha, "topk": topk, "smooth": sm,
                                  **{f"VAL_{k}": v for k, v in vm.items()}})
                key = (crit, vm["F1"])
                if val_has_anom and (best is None or key > best[0]):
                    best = (key, alpha, topk, sm)
    if best is None:                     # no labelled validation anomalies
        best = ((np.nan, np.nan), alpha_grid[0], CFG["TOPK_GRID"][0],
                CFG["SMOOTH_GRID"][0])
    _, ALPHA, TOPK, SMOOTH = best
    print(f"[SELECT] unsup: alpha={ALPHA} top-k={TOPK} smooth={SMOOTH} "
          f"(criterion: VAL {CFG['SELECT_BY']})")
    unsup_raw_sel = unsup_raw(ALPHA, TOPK)

    # ---- residual-space classifier --------------------------------------
    # Descriptors are built from the SAME residual matrices the unsupervised
    # scorer uses. The classifier is fitted on CALFIT (normal, never seen by
    # the forecasting network) against TRANOM (the train split's own anomalous
    # windows). Validation supplies the model-selection criterion only.
    def descriptors(split):
        parts = [residual_features(streams[name][split])
                 for name in stream_slices]
        parts = [p for p in parts if p.shape[1] > 0]
        if not parts:
            return np.zeros((START_OF[split].shape[0], 0), dtype=np.float32)
        return np.concatenate(parts, axis=1)

    n_tranom = int(tra_starts.size)
    head = head_params = None
    head_val_ap = float("nan")
    want_supervised = any(s in CFG["SCORERS"] for s in ("semisup", "hybrid"))
    if want_supervised and val_has_anom and n_tranom >= CFG["MIN_TRAIN_ANOM"]:
        D_cf, D_ta = descriptors("calfit"), descriptors("tranom")
        X_head = np.concatenate([D_cf, D_ta], axis=0)
        y_head = np.concatenate([np.zeros(D_cf.shape[0], dtype=int),
                                 np.ones(D_ta.shape[0], dtype=int)])
        head, head_params, head_val_ap = fit_supervised_head(
            X_head, y_head, descriptors("val"), yva, seed)
        del D_cf, D_ta, X_head, y_head
    elif want_supervised:
        print(f"[WARN] residual classifier skipped: {n_tranom} anomalous train "
              f"windows (need {CFG['MIN_TRAIN_ANOM']}) or no validation "
              f"anomalies. Only the unsupervised scorer is reported.")
    if head is not None:
        print(f"[SELECT] residual classifier {head_params} "
              f"(VAL PR-AUC {head_val_ap:.4f}, {n_tranom} train positives)")

    head_raw = None
    if head is not None:
        head_raw = {sp: logit(head.predict_proba(descriptors(sp))[:, 1])
                    for sp in EVAL}
        mu_h, md_h = fit_mad(head_raw["calthr"])
        head_raw = {sp: apply_mad(head_raw[sp], mu_h, md_h) for sp in EVAL}

    # ---- assemble the scorers -------------------------------------------
    # Every scorer selects its own smoothing length (and, for the hybrid, its
    # mixing weight) on validation, so no scorer is handicapped by another's
    # choices and the ablation between them is fair.
    def select_smooth(raw, tag, extra=None):
        best_local, rows_ = None, []
        for sm in CFG["SMOOTH_GRID"]:
            sc = smoothed(raw, sm)
            thr = threshold_from_normals(
                np.concatenate([sc["calthr"], sc["val"][yva == 0]]),
                CFG["TARGET_FPR"])
            vm = compute_metrics(yva, (sc["val"] > thr).astype(int), sc["val"])
            crit = vm.get(CFG["SELECT_BY"], np.nan)
            crit = -np.inf if not np.isfinite(crit) else crit
            rows_.append({"mode": mode, "seed": seed, "scorer": tag,
                          "smooth": sm, **(extra or {}),
                          **{f"VAL_{k}": v for k, v in vm.items()}})
            if val_has_anom and (best_local is None or (crit, vm["F1"]) > best_local[0]):
                best_local = ((crit, vm["F1"]), sm)
        return (best_local[1] if best_local else CFG["SMOOTH_GRID"][0]), rows_

    scorer_specs = []                     # (tag, raw dict, params dict)
    if "unsup" in CFG["SCORERS"]:
        scorer_specs.append(("unsup", unsup_raw_sel,
                             {"alpha": ALPHA, "topk": TOPK, "smooth": SMOOTH,
                              "beta": 0.0}))
    if head is not None and "semisup" in CFG["SCORERS"]:
        sm_ss, rows_ss = select_smooth(head_raw, "semisup")
        grid_rows.extend(rows_ss)
        scorer_specs.append(("semisup", head_raw,
                             {"alpha": np.nan, "topk": np.nan,
                              "smooth": sm_ss, "beta": 1.0}))
    if head is not None and "hybrid" in CFG["SCORERS"]:
        best_hy, rows_hy = None, []
        for beta in CFG["BETA_GRID"]:
            mix = {sp: (1.0 - beta) * unsup_raw_sel[sp] + beta * head_raw[sp]
                   for sp in EVAL}
            sm_b, rows_b = select_smooth(mix, "hybrid", {"beta": beta})
            rows_hy.extend(rows_b)
            crit = max((r[f"VAL_{CFG['SELECT_BY']}"] for r in rows_b
                        if np.isfinite(r.get(f"VAL_{CFG['SELECT_BY']}", np.nan))),
                       default=-np.inf)
            if best_hy is None or crit > best_hy[0]:
                best_hy = (crit, beta, sm_b, mix)
        grid_rows.extend(rows_hy)
        if best_hy is not None:
            _, BETA, sm_hy, mix_sel = best_hy
            print(f"[SELECT] hybrid: beta={BETA} smooth={sm_hy}")
            scorer_specs.append(("hybrid", mix_sel,
                                 {"alpha": ALPHA, "topk": TOPK,
                                  "smooth": sm_hy, "beta": BETA}))

    # ---- evaluate every scorer at every operating point ------------------
    rows, op_rows = [], []
    fig_payload = None
    for tag, raw, params in scorer_specs:
        sc = smoothed(raw, params["smooth"])
        c_s, v_s, t_s = sc["calthr"], sc["val"], sc["test"]
        norm_pool = np.concatenate([c_s, v_s[yva == 0]])

        thr_fpr = threshold_from_normals(norm_pool, CFG["TARGET_FPR"])
        if val_has_anom:
            thr_f1, val_best_f1 = best_threshold(yva, v_s, "F1")
            thr_acc, val_best_acc = best_threshold(yva, v_s, "Accuracy")
        else:
            thr_f1, val_best_f1 = thr_fpr, np.nan
            thr_acc, val_best_acc = thr_fpr, np.nan

        # hysteresis ratio selected on validation F1
        best_h = None
        for mult in CFG["HYST_GRID"]:
            lo_thr = threshold_from_normals(
                norm_pool, min(0.5, CFG["TARGET_FPR"] * mult))
            vp = hysteresis_predict(va_starts, v_s, S, thr_f1, lo_thr)
            f1_h = compute_metrics(yva, vp)["F1"] if val_has_anom else np.nan
            if val_has_anom and (best_h is None or f1_h > best_h[0]):
                best_h = (f1_h, mult, lo_thr)
        HYST, thr_lo = (best_h[1], best_h[2]) if best_h else (1, thr_f1)

        pred_fpr = (t_s > thr_fpr).astype(int)
        pred_f1 = (t_s > thr_f1).astype(int)
        pred_acc = (t_s > thr_acc).astype(int)
        pred_hy = hysteresis_predict(te_starts, t_s, S, thr_f1, thr_lo)

        val_m = compute_metrics(yva, (v_s > thr_fpr).astype(int), v_s)
        test_m = compute_metrics(yte, pred_fpr, t_s)
        test_op = compute_metrics(yte, pred_f1, t_s)
        test_ac = compute_metrics(yte, pred_acc, t_s)
        test_hy = compute_metrics(yte, pred_hy, t_s)
        ev_fpr = event_metrics(te_starts, yte, pred_fpr, S)
        ev_op = event_metrics(te_starts, yte, pred_f1, S)
        ev_ac = event_metrics(te_starts, yte, pred_acc, S)
        ev_hy = event_metrics(te_starts, yte, pred_hy, S)

        cis = {
            "recall_wilson95": wilson_ci(test_m["TP"], test_m["TP"] + test_m["FN"]),
            "precision_wilson95": wilson_ci(test_m["TP"], test_m["TP"] + test_m["FP"]),
            "rocauc_boot95": bootstrap_ci(yte, t_s, safe_auc, CFG["N_BOOT"], seed),
            "prauc_boot95": bootstrap_ci(yte, t_s, safe_ap, CFG["N_BOOT"], seed),
        }

        # full false-positive-budget trade-off curve for this scorer
        for f in CFG["FPR_SWEEP"]:
            th = threshold_from_normals(norm_pool, f)
            m_ = compute_metrics(yte, (t_s > th).astype(int), t_s)
            op_rows.append({"mode": mode, "seed": seed, "scorer": tag,
                            "target_FPR": f, "threshold": th,
                            **{f"TEST_{k}": v for k, v in m_.items()}})

        rows.append({
            "mode": mode, "seed": seed, "scorer": tag,
            "alpha": params["alpha"], "topk": params["topk"],
            "smooth": params["smooth"], "beta": params["beta"],
            "hyst_mult": HYST,
            "head": json.dumps(head_params) if head_params else "",
            "head_val_PRAUC": head_val_ap,
            "thr_fpr": thr_fpr, "thr_valF1": thr_f1, "thr_valAcc": thr_acc,
            "thr_hyst_lo": thr_lo,
            "val_best_F1": val_best_f1, "val_best_Accuracy": val_best_acc,
            **{f"VAL_{k}": v for k, v in val_m.items()},
            **{f"TEST_{k}": v for k, v in test_m.items()},
            **{f"TESTOP_{k}": v for k, v in test_op.items()},
            **{f"TESTACC_{k}": v for k, v in test_ac.items()},
            **{f"TESTHY_{k}": v for k, v in test_hy.items()},
            **{f"TEST_{k}": v for k, v in ev_fpr.items()},
            **{f"TESTOP_{k}": v for k, v in ev_op.items()},
            **{f"TESTACC_{k}": v for k, v in ev_ac.items()},
            **{f"TESTHY_{k}": v for k, v in ev_hy.items()},
            **{f"CI_{k}_lo": v[0] for k, v in cis.items()},
            **{f"CI_{k}_hi": v[1] for k, v in cis.items()},
            "n_train_anom_windows": n_tranom,
            **split_doc})

        if make_figures and fig_payload is None:
            fig_payload = (tag, t_s, thr_fpr, pred_fpr)

    # ---- one-factor-at-a-time sensitivity around the selected point ------
    # The sweep is reported on VALIDATION, never on test, so it documents how
    # sharp the selection was without turning the test split into a tuning set.
    sens_alpha = sens_topk = sens_smooth = None
    if make_figures and val_has_anom:
        def _sweep(param, values):
            out = []
            for v in values:
                a, k, s_ = ALPHA, TOPK, SMOOTH
                if param == "alpha":
                    a = v
                elif param == "topk":
                    k = v
                else:
                    s_ = v
                sc = smoothed(unsup_raw(a, k), s_)
                th = threshold_from_normals(
                    np.concatenate([sc["calthr"], sc["val"][yva == 0]]),
                    CFG["TARGET_FPR"])
                m_ = compute_metrics(yva, (sc["val"] > th).astype(int), sc["val"])
                out.append({param: v, "alpha": a, "topk": k, "smooth": s_,
                            "scorer": "unsup", "sweep_split": "VAL",
                            **{f"VAL_{kk}": vvv for kk, vvv in m_.items()}})
            return out
        if mode == "fused":
            sens_alpha = _sweep("alpha", CFG["ALPHA_GRID"])
        sens_topk = _sweep("topk", CFG["TOPK_GRID"])
        sens_smooth = _sweep("smooth", CFG["SMOOTH_GRID"])

    # ---- figures from this run only --------------------------------------
    if make_figures:
        save_learning_curve(hist.history, mode, seed, result_dir)
    if make_figures and fig_payload is not None:
        f_tag, t_s_fig, thr_fig, pred_fig = fig_payload
        save_detection_figures(yte, t_s_fig, thr_fig, pred_fig, op_rows,
                               mode, f_tag, result_dir)

    # ---- free memory before the next run ---------------------------------
    try:
        hist.model = None       # Keras 2: break the History -> model reference
    except AttributeError:
        pass                    # Keras 3: read-only property; del below suffices
    del hist
    del model, PRED, streams, _agg_cache
    del train_seq, val_seq
    del T_cf, T_ct, T_ta, T_va, T_te
    del data
    tf.keras.backend.clear_session()
    gc.collect()
    mem("after cleanup")

    return rows, grid_rows, op_rows, sens_alpha, sens_topk, sens_smooth
