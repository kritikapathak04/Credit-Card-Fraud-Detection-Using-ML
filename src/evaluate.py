"""Model evaluation, threshold tuning, and plotting utilities."""
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def tune_threshold(y_true: np.ndarray, y_probs: np.ndarray) -> float:
    """Find the decision threshold that maximizes F1-score on validation data."""
    thresholds = np.linspace(0.01, 0.95, 190)
    best_thr = 0.5
    best_f1 = -1.0

    for thr in thresholds:
        preds = (y_probs >= thr).astype(int)
        # Avoid zero division warnings
        score = f1_score(y_true, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thr = float(thr)

    return round(best_thr, 3)


def compute_metrics(
    y_true: np.ndarray, y_probs: np.ndarray, threshold: float = 0.5
) -> Dict[str, float]:
    """Calculate key evaluation metrics suited for rare-event imbalanced data."""
    y_pred = (y_probs >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # PR-AUC and ROC-AUC are threshold-independent
    try:
        pr_auc = float(average_precision_score(y_true, y_probs))
    except Exception:
        pr_auc = 0.0

    try:
        roc_auc = float(roc_auc_score(y_true, y_probs))
    except Exception:
        roc_auc = 0.5

    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    mcc = float(matthews_corrcoef(y_true, y_pred))

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "pr_auc": round(pr_auc, 4),
        "roc_auc": round(roc_auc, 4),
        "mcc": round(mcc, 4),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "threshold": round(threshold, 3),
    }


def plot_roc_curves(
    curves_data: Dict[str, Tuple[np.ndarray, np.ndarray, float]], save_path: Path
):
    """Plot ROC curves for all models and save figure."""
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#2563EB", "#0D9488", "#F59E0B", "#E11D48"]
    for i, (name, (fpr, tpr, score)) in enumerate(curves_data.items()):
        color = colors[i % len(colors)]
        ax.plot(fpr, tpr, label=f"{name} (AUC = {score:.3f})", color=color, linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Chance (0.500)")
    ax.set_title("Receiver Operating Characteristic (ROC) Curves", fontsize=12, fontweight="bold")
    ax.set_xlabel("False Positive Rate", fontsize=10)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=10)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_pr_curves(
    curves_data: Dict[str, Tuple[np.ndarray, np.ndarray, float]],
    baseline_rate: float,
    save_path: Path,
):
    """Plot Precision-Recall curves for all models."""
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#2563EB", "#0D9488", "#F59E0B", "#E11D48"]
    for i, (name, (prec, rec, score)) in enumerate(curves_data.items()):
        color = colors[i % len(colors)]
        ax.plot(rec, prec, label=f"{name} (PR-AUC = {score:.3f})", color=color, linewidth=2)

    ax.axhline(
        y=baseline_rate,
        color="k",
        linestyle="--",
        alpha=0.5,
        label=f"Baseline ({baseline_rate:.3%})",
    )
    ax.set_title("Precision-Recall (PR) Curves", fontsize=12, fontweight="bold")
    ax.set_xlabel("Recall", fontsize=10)
    ax.set_ylabel("Precision", fontsize=10)
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_confusion_matrix(cm_counts: Dict[str, int], title: str, save_path: Path):
    """Plot annotated 2x2 confusion matrix."""
    tn = cm_counts["tn"]
    fp = cm_counts["fp"]
    fn = cm_counts["fn"]
    tp = cm_counts["tp"]
    matrix = np.array([[tn, fp], [fn, tp]])

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    cax = ax.matshow(matrix, cmap="Blues", alpha=0.7)
    fig.colorbar(cax)

    labels = [
        [f"TN\n{tn:,}", f"FP (False Alarm)\n{fp:,}"],
        [f"FN (Missed Fraud)\n{fn:,}", f"TP (Caught Fraud)\n{tp:,}"],
    ]

    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                labels[i][j],
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold",
                color="black" if matrix[i, j] < matrix.max() * 0.7 else "white",
            )

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted Legit (0)", "Predicted Fraud (1)"])
    ax.set_yticklabels(["Actual Legit (0)", "Actual Fraud (1)"])
    ax.set_title(f"Confusion Matrix: {title}", fontsize=11, fontweight="bold", pad=15)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_feature_importance(
    importance_df: pd.DataFrame, top_n: int, save_path: Path
):
    """Plot top N most influential features."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    top = importance_df.head(top_n).iloc[::-1]
    ax.barh(top["feature"], top["importance"], color="#0D9488", edgecolor="#115E59")
    ax.set_title("Top Features Influencing Fraud Detection", fontsize=12, fontweight="bold")
    ax.set_xlabel("Relative Importance Score", fontsize=10)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_class_distribution(y: pd.Series, save_path: Path):
    """Plot log-scaled class distribution."""
    fig, ax = plt.subplots(figsize=(5.5, 4))
    counts = y.value_counts().sort_index()
    bars = ax.bar(["Legitimate (0)", "Fraud (1)"], counts.values, color=["#10B981", "#EF4444"])
    ax.set_yscale("log")
    ax.set_title("Transaction Class Distribution (Log Scale)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Number of Transactions (Log)", fontsize=10)

    for bar in bars:
        h = bar.get_height()
        ax.annotate(
            f"{int(h):,}",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
