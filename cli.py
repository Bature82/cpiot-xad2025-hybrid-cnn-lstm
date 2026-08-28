"""Main loop: ablation x seeds, resumable, plus the export stage.

Every finished run is written to ``RESULT_DIR/partial/`` immediately. If the
session dies, re-running the same command reloads the completed runs and
continues from the first unfinished one.
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from .config import CFG
from .data import load_corpus
from .protocol import run_protocol
from .utils import jsonable, mem


def build_parser():
    p = argparse.ArgumentParser(
        prog="cpiot-xad",
        description="CPIoT-XAD2025: hybrid CNN/LSTM cross-domain anomaly "
                    "detection under a leakage-safe protocol.")
    p.add_argument("--data-path", required=True,
                   help="Path to the fused corpus CSV.")
    p.add_argument("--result-dir", default="results",
                   help="Directory for metrics, figures and checkpoints.")
    p.add_argument("--modes", nargs="+", default=None,
                   choices=["fused", "net_only", "sen_only"],
                   help="Ablation modes to run (default: all three).")
    p.add_argument("--seeds", nargs="+", type=int, default=None,
                   help="Seeds to repeat the protocol over.")
    p.add_argument("--scorers", nargs="+", default=None,
                   choices=["unsup", "semisup", "hybrid"],
                   help="Scoring rules to build and report.")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--window-size", type=int, default=None)
    p.add_argument("--stride", type=int, default=None)
    p.add_argument("--target-fpr", type=float, default=None,
                   help="False-positive budget for the primary threshold.")
    p.add_argument("--select-by", default=None,
                   choices=["PR-AUC", "ROC-AUC", "F1"],
                   help="Validation criterion used for model selection.")
    p.add_argument("--label-col", default=None,
                   help="Force a specific label column instead of auto-detecting.")
    p.add_argument("--net-suffix", default=None,
                   help="Column suffix identifying the network stream.")
    p.add_argument("--sen-suffix", default=None,
                   help="Column suffix identifying the process/sensor stream.")
    p.add_argument("--n-boot", type=int, default=None,
                   help="Bootstrap resamples for the AUC confidence intervals.")
    return p


def apply_overrides(args):
    """Fold command-line arguments into the global CFG."""
    CFG["DATA_PATH"] = args.data_path
    CFG["RESULT_DIR"] = args.result_dir
    for key, val in [
        ("MODES", args.modes), ("SEEDS", args.seeds), ("SCORERS", args.scorers),
        ("EPOCHS", args.epochs), ("BATCH_SIZE", args.batch_size),
        ("WINDOW_SIZE", args.window_size), ("STRIDE", args.stride),
        ("TARGET_FPR", args.target_fpr), ("SELECT_BY", args.select_by),
        ("FORCE_LABEL_COL", args.label_col), ("NET_SUFFIX", args.net_suffix),
        ("SEN_SUFFIX", args.sen_suffix), ("N_BOOT", args.n_boot),
    ]:
        if val is not None:
            CFG[key] = val
    os.makedirs(CFG["RESULT_DIR"], exist_ok=True)


def export(all_rows, all_grid, all_ops, n_net, n_sen, t0, primary_seed):
    """Write every reported number into one set of files."""
    result_dir = CFG["RESULT_DIR"]
    res = pd.DataFrame(all_rows)
    res.to_csv(os.path.join(result_dir, "metrics_all_runs.csv"), index=False)

    metric_cols = [c for c in res.columns
                   if c.startswith(("VAL_", "TEST_", "TESTOP_", "TESTACC_", "TESTHY_"))
                   and res[c].dtype != object]
    group_keys = ["mode", "scorer"] if "scorer" in res.columns else ["mode"]
    summary = (res.groupby(group_keys)[metric_cols]
               .agg(["mean", "std", "min", "max", "median"]).round(4))
    summary.to_csv(os.path.join(result_dir, "metrics_summary.csv"))

    print("\n============== SUMMARY: mean +/- std over seeds ==============")
    mean_show = [(c, "mean") for c in
                 ["TEST_Accuracy", "TEST_Precision", "TEST_Recall", "TEST_F1",
                  "TEST_ROC-AUC", "TEST_PR-AUC", "TEST_EventRecall"]]
    print(summary[[c for c in mean_show if c in summary.columns]])

    print("\n============== SUMMARY: BEST seed per configuration ==============")
    best_show = [(c, "max") for c in
                 ["TEST_Accuracy", "TESTACC_Accuracy", "TESTOP_F1", "TESTHY_F1",
                  "TEST_F1", "TEST_ROC-AUC", "TEST_PR-AUC"]]
    print(summary[[c for c in best_show if c in summary.columns]])

    # ---- the single best run per (mode, scorer), by several criteria -----
    best_records = []
    BEST_CRITERIA = ["TEST_Accuracy", "TESTACC_Accuracy", "TESTOP_Accuracy",
                     "TESTHY_Accuracy", "TEST_F1", "TESTOP_F1", "TESTACC_F1",
                     "TESTHY_F1", "TEST_ROC-AUC", "TEST_PR-AUC"]
    report_cols = [c for c in
                   ["mode", "scorer", "seed", "alpha", "topk", "smooth", "beta",
                    "hyst_mult", "head", "TEST_Accuracy", "TEST_Precision",
                    "TEST_Recall", "TEST_F1", "TEST_ROC-AUC", "TEST_PR-AUC",
                    "TESTOP_Accuracy", "TESTOP_F1", "TESTACC_Accuracy",
                    "TESTACC_F1", "TESTHY_Accuracy", "TESTHY_F1",
                    "TEST_EventRecall"] if c in res.columns]
    for keys, grp in res.groupby(group_keys):
        keys = keys if isinstance(keys, tuple) else (keys,)
        for crit in BEST_CRITERIA:
            if crit not in grp.columns:
                continue
            col = pd.to_numeric(grp[crit], errors="coerce")
            if not np.isfinite(col).any():
                continue
            grp = grp.assign(**{crit: col})
            row_best = grp.loc[grp[crit].idxmax()]
            best_records.append({"criterion": crit,
                                 "criterion_value": float(row_best[crit]),
                                 **{k: row_best[k] for k in report_cols}})
    if best_records:
        best_df = pd.DataFrame(best_records)
        best_df.to_csv(os.path.join(result_dir, "best_runs.csv"), index=False)
        print("\n============== BEST SINGLE RUNS (by test accuracy) ==============")
        acc_best = best_df[best_df["criterion"] == "TEST_Accuracy"]
        cols = [c for c in ["mode", "scorer", "seed", "TEST_Accuracy",
                            "TEST_Precision", "TEST_Recall", "TEST_F1"]
                if c in acc_best.columns]
        if len(acc_best):
            print(acc_best[cols].round(4).to_string(index=False))

    # ---- headline: the best configuration anywhere in the study ----------
    for crit in ("TEST_Accuracy", "TESTACC_Accuracy", "TESTHY_Accuracy"):
        if crit not in res.columns:
            continue
        col = pd.to_numeric(res[crit], errors="coerce")
        if np.isfinite(col).any():
            b = res.loc[col.idxmax()]
            print(f"\n[BEST {crit}] {b.get('mode')}/{b.get('scorer')} seed "
                  f"{b.get('seed')}: accuracy {b[crit]:.4f}, "
                  f"F1 {b.get(crit.replace('Accuracy', 'F1'), float('nan')):.4f}, "
                  f"ROC-AUC {b.get('TEST_ROC-AUC', float('nan')):.4f}")

    if all_grid:
        pd.DataFrame(all_grid).to_csv(
            os.path.join(result_dir, "selection_grid.csv"), index=False)
    if all_ops:
        ops_df = pd.DataFrame(all_ops)
        ops_df.to_csv(os.path.join(result_dir, "operating_points.csv"), index=False)
        op_metric_cols = [c for c in ops_df.columns
                          if c.startswith("TEST_") and ops_df[c].dtype != object]
        (ops_df.groupby([k for k in group_keys] + ["target_FPR"])[op_metric_cols]
         .agg(["mean", "std", "max"]).round(4)
         .to_csv(os.path.join(result_dir, "operating_points_summary.csv")))

    primary_rows = res[(res["mode"] == "fused") & (res["seed"] == primary_seed)]
    if len(primary_rows):
        primary = primary_rows.iloc[0]
        cis_out = {k: [primary.get(f"CI_{k}_lo"), primary.get(f"CI_{k}_hi")]
                   for k in ["recall_wilson95", "precision_wilson95",
                             "rocauc_boot95", "prauc_boot95"]}
        with open(os.path.join(result_dir, "cis.json"), "w") as f:
            json.dump(cis_out, f, indent=2)

    with open(os.path.join(result_dir, "run_config.json"), "w") as f:
        json.dump({**CFG,
                   "scaler": "StandardScaler (z-score), fit on TRAIN-split normal "
                             "rows only",
                   "imputation": "per-feature median, fit on TRAIN-split rows only",
                   "calibration": "residual median/MAD, score standardisation and the "
                                  "classifier's negative class come from CALFIT; every "
                                  "decision threshold comes from CALTHR pooled with "
                                  "VALIDATION normals. The two are disjoint subsets of "
                                  "held-out TRAIN normal windows.",
                   "supervision": "the forecasting network is fitted on TRAIN normal "
                                  "windows only; the residual-space classifier, where "
                                  "used, is fitted on TRAIN windows only (CALFIT normals "
                                  "as negatives, TRAIN anomalies as positives)",
                   "selection": "alpha, top-k, smoothing, classifier hyper-parameters, "
                                "the hybrid mixing weight and the hysteresis ratio are "
                                f"all chosen by validation {CFG['SELECT_BY']} or "
                                "validation F1; test is used once for reporting",
                   "n_net_features": n_net, "n_sensor_features": n_sen,
                   "runtime_sec": round(time.time() - t0, 1)}, f, indent=2)

    print("\n[Done] Reported results come from:")
    for fn in ["metrics_all_runs.csv", "metrics_summary.csv", "best_runs.csv",
               "operating_points.csv", "operating_points_summary.csv", "cis.json",
               "selection_grid.csv", "sensitivity_alpha.csv", "sensitivity_topk.csv",
               "sensitivity_smooth.csv", "run_config.json"]:
        print("   -", os.path.join(result_dir, fn))


def main(argv=None):
    args = build_parser().parse_args(argv)
    apply_overrides(args)

    X_np, y_rows, nb_cols, sw_cols = load_corpus(CFG["DATA_PATH"])
    n_net, n_sen = len(nb_cols), len(sw_cols)
    mem("after data load")

    t0 = time.time()
    all_rows, all_grid, all_ops = [], [], []
    primary_seed = CFG["SEEDS"][0]
    result_dir = CFG["RESULT_DIR"]
    partial_dir = os.path.join(result_dir, "partial")
    os.makedirs(partial_dir, exist_ok=True)

    for mode in CFG["MODES"]:
        for seed in CFG["SEEDS"]:
            row_path = os.path.join(partial_dir, f"row_{mode}_seed{seed}.json")
            op_path = os.path.join(partial_dir, f"ops_{mode}_seed{seed}.json")
            if os.path.exists(row_path):
                with open(row_path) as f:
                    cached = json.load(f)
                all_rows.extend(cached if isinstance(cached, list) else [cached])
                if os.path.exists(op_path):
                    with open(op_path) as f:
                        all_ops.extend(json.load(f))
                print(f"[RESUME] {mode}/seed{seed} completed in an earlier session "
                      f"- loaded from disk, skipping.")
                continue

            make_figs = (seed == primary_seed)
            print(f"\n########## mode={mode} seed={seed} ##########")
            t_run = time.time()
            run_rows, grid, ops, sa, sk, ss = run_protocol(
                mode, seed, X_np, y_rows, n_net, n_sen, make_figures=make_figs)
            all_rows.extend(run_rows)
            all_grid.extend(grid)
            all_ops.extend(ops)
            for payload, fname in ((sa, "sensitivity_alpha.csv"),
                                   (sk, "sensitivity_topk.csv"),
                                   (ss, "sensitivity_smooth.csv")):
                if payload is not None:
                    pd.DataFrame(payload).to_csv(
                        os.path.join(result_dir, fname), index=False)

            with open(row_path, "w") as f:
                json.dump([jsonable(r) for r in run_rows], f)
            with open(op_path, "w") as f:
                json.dump([jsonable(r) for r in ops], f)
            pd.DataFrame(grid).to_csv(
                os.path.join(partial_dir, f"grid_{mode}_seed{seed}.csv"), index=False)

            for r in run_rows:
                print({"scorer": r["scorer"],
                       **{k: (round(v, 4) if isinstance(v, float) else v)
                          for k, v in r.items()
                          if k.startswith(("TEST_", "TESTACC_")) and
                          k.split("_", 1)[1] in ("Accuracy", "Precision", "Recall",
                                                 "F1", "ROC-AUC", "PR-AUC")}})
            print(f"[TIME] {mode}/seed{seed} finished in "
                  f"{(time.time() - t_run) / 60:.1f} min "
                  f"({time.time() - t0:.0f}s total this session)")

    export(all_rows, all_grid, all_ops, n_net, n_sen, t0, primary_seed)


if __name__ == "__main__":
    main()
