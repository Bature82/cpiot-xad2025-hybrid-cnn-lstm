"""Figures for the primary seed of each ablation mode.

Every figure illustrates the FIRST scorer that produced a row, so the picture
and the headline table always describe the same scoring rule.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    auc, confusion_matrix, precision_recall_curve, roc_curve,
)

from .metrics import safe_ap  # noqa: E402


def save_learning_curve(history, mode, seed, result_dir):
    hd = pd.DataFrame(history)
    hd.to_csv(os.path.join(result_dir, f"history_{mode}_seed{seed}.csv"),
              index=False)
    plt.figure()
    plt.plot(hd["loss"], label="training loss (normal windows)")
    if "val_loss" in hd:
        plt.plot(hd["val_loss"], label="validation loss (normal windows)")
    plt.xlabel("Epoch")
    plt.ylabel("One-step forecast loss (Huber)")
    plt.legend()
    plt.title(f"Learning curve ({mode})")
    plt.savefig(os.path.join(result_dir, f"loss_{mode}.png"),
                dpi=150, bbox_inches="tight")
    plt.close()


def save_detection_figures(yte, t_s, thr_fpr, pred_fpr, op_rows, mode, tag,
                           result_dir):
    """ROC, precision-recall, confusion matrix, score histogram and the
    false-alarm budget trade-off curve for one scorer on the test split."""
    fpr, tpr, _ = roc_curve(yte, t_s)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC-AUC = {auc(fpr, tpr):.4f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve (Test, {mode}, {tag})")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(result_dir, f"roc_{mode}.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    pr_p, pr_r, _ = precision_recall_curve(yte, t_s)
    plt.figure(figsize=(6, 5))
    plt.plot(pr_r, pr_p, label=f"PR-AUC (AP) = {safe_ap(yte, t_s):.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve (Test, {mode}, {tag})")
    plt.legend(loc="upper right")
    plt.savefig(os.path.join(result_dir, f"pr_{mode}.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    cm = confusion_matrix(yte, pred_fpr, labels=[0, 1])
    plt.figure(figsize=(4.5, 4))
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"Confusion Matrix (Test, {mode}, {tag})")
    plt.xticks([0, 1], ["Pred Normal", "Pred Anomaly"])
    plt.yticks([0, 1], ["True Normal", "True Anomaly"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.savefig(os.path.join(result_dir, f"cm_{mode}.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.hist(t_s[np.asarray(yte) == 0], bins=80, alpha=0.6, density=True,
             label="normal")
    plt.hist(t_s[np.asarray(yte) == 1], bins=80, alpha=0.6, density=True,
             label="anomalous")
    plt.axvline(thr_fpr, color="k", ls="--", label="threshold (target FPR)")
    plt.xlabel("Anomaly score")
    plt.ylabel("Density")
    plt.title(f"Test score distribution ({mode}, {tag})")
    plt.legend()
    plt.savefig(os.path.join(result_dir, f"scores_{mode}.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    oc = [r for r in op_rows if r["scorer"] == tag]
    if oc:
        plt.figure(figsize=(6, 4.5))
        plt.plot([r["target_FPR"] for r in oc],
                 [r["TEST_Recall"] for r in oc], "o-", label="Recall")
        plt.plot([r["target_FPR"] for r in oc],
                 [r["TEST_Precision"] for r in oc], "s-", label="Precision")
        plt.plot([r["target_FPR"] for r in oc],
                 [r["TEST_F1"] for r in oc], "^-", label="F1")
        plt.xscale("log")
        plt.xlabel("Target false-positive rate on normal windows")
        plt.ylabel("Test score")
        plt.title(f"False-alarm budget trade-off ({mode}, {tag})")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(result_dir, f"opcurve_{mode}.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()
