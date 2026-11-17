"""Main training orchestrator for model comparison, threshold tuning, and artifact generation."""
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.config import (
    ARTIFACT_PATH,
    MODELS_DIR,
    RANDOM_STATE,
    RESULTS_DIR,
    TARGET,
    TEST_SIZE,
    VAL_SIZE,
)
from src.evaluate import (
    compute_metrics,
    plot_class_distribution,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_pr_curves,
    plot_roc_curves,
    tune_threshold,
)
from src.models import get_candidate_models
from src.preprocessing import (
    build_preprocessor,
    clean_data,
    engineer_features,
    get_feature_matrix_and_target,
)


def run_training(
    df: pd.DataFrame,
    fast: bool = False,
    source: str = "creditcard.csv",
    verbose: bool = True,
) -> Dict[str, Any]:
    """Execute complete end-to-end training pipeline.

    1. Cleaning & feature engineering
    2. Stratified train (64%) / val (16%) / test (20%) split
    3. Train 4 candidate models with class weighting
    4. Validation selection by PR-AUC & validation threshold tuning
    5. Final evaluation on untouched test set
    6. Save models/best_model.joblib and results/ artifacts
    """
    start_total = time.time()
    if verbose:
        print("=" * 65)
        print("🚀 Starting Credit Card Fraud Detection Training Pipeline")
        print("=" * 65)

    # 1. Clean data and plot class distribution
    df_clean = clean_data(df)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    plot_class_distribution(df_clean[TARGET], RESULTS_DIR / "class_distribution.png")

    # 2. Extract features & target
    X, y = get_feature_matrix_and_target(df_clean)
    if verbose:
        n_fraud = int(y.sum())
        print(f"Dataset: {len(X):,} transactions | {n_fraud:,} frauds ({100*n_fraud/len(X):.3f}%)")

    # 3. Stratified splits: 64% train / 16% val / 20% test
    # First split out test set (20%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    # Then split temp into train (80% of 80% = 64%) and val (20% of 80% = 16%)
    val_fraction_of_temp = VAL_SIZE / (1.0 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_fraction_of_temp, stratify=y_temp, random_state=RANDOM_STATE
    )

    if verbose:
        print(f"Splits -> Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

    # 4. Train candidates and evaluate on validation set
    candidate_estimators = get_candidate_models(fast=fast, random_state=RANDOM_STATE)
    trained_pipelines = {}
    val_results = {}
    train_times = {}

    for name, estimator in candidate_estimators.items():
        if verbose:
            print(f"\n--- Training {name} ---")
        t0 = time.time()
        preprocessor = build_preprocessor()
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", estimator),
        ])
        pipe.fit(X_train, y_train)
        elapsed = time.time() - t0
        train_times[name] = elapsed

        # Predict validation probabilities
        val_probs = pipe.predict_proba(X_val)[:, 1]
        val_metrics = compute_metrics(y_val.to_numpy(), val_probs, threshold=0.5)
        val_results[name] = {
            "val_pr_auc": val_metrics["pr_auc"],
            "val_roc_auc": val_metrics["roc_auc"],
            "val_probs": val_probs,
        }
        trained_pipelines[name] = pipe
        if verbose:
            print(f"    Val PR-AUC: {val_metrics['pr_auc']:.4f} | ROC-AUC: {val_metrics['roc_auc']:.4f} ({elapsed:.1f}s)")

    # 5. Model Selection based on validation PR-AUC
    best_name = max(val_results, key=lambda k: val_results[k]["val_pr_auc"])
    best_pipe = trained_pipelines[best_name]

    # Tune threshold specifically for champion model on validation set
    best_threshold = tune_threshold(y_val.to_numpy(), val_results[best_name]["val_probs"])

    if verbose:
        print(f"\n⭐ Best Model Selected: {best_name}")
        print(f"⭐ Optimal Decision Threshold tuned on validation set: {best_threshold:.3f}")

    # 6. Evaluation on the Untouched Test Set
    roc_curves_data = {}
    pr_curves_data = {}
    comparison_rows = []

    for name, pipe in trained_pipelines.items():
        test_probs = pipe.predict_proba(X_test)[:, 1]
        # Use optimal threshold for best model, 0.5 for others (or tuned threshold)
        thr = best_threshold if name == best_name else 0.5
        t_metrics = compute_metrics(y_test.to_numpy(), test_probs, threshold=thr)

        # ROC Curve data
        fpr, tpr, _ = roc_curve(y_test, test_probs)
        roc_curves_data[name] = (fpr, tpr, t_metrics["roc_auc"])

        # PR Curve data
        prec, rec, _ = precision_recall_curve(y_test, test_probs)
        pr_curves_data[name] = (prec, rec, t_metrics["pr_auc"])

        comparison_rows.append({
            "model": name,
            "val_pr_auc": val_results[name]["val_pr_auc"],
            "test_precision": t_metrics["precision"],
            "test_recall": t_metrics["recall"],
            "test_f1": t_metrics["f1"],
            "test_pr_auc": t_metrics["pr_auc"],
            "test_roc_auc": t_metrics["roc_auc"],
            "test_mcc": t_metrics["mcc"],
            "train_seconds": round(train_times[name], 2),
        })

    comp_df = pd.DataFrame(comparison_rows).sort_values("test_pr_auc", ascending=False)
    comp_df.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)

    # 7. Generate Evaluation Plots
    plot_roc_curves(roc_curves_data, RESULTS_DIR / "roc_curves.png")
    test_baseline = float(y_test.mean())
    plot_pr_curves(pr_curves_data, baseline_rate=test_baseline, save_path=RESULTS_DIR / "pr_curves.png")

    best_test_probs = best_pipe.predict_proba(X_test)[:, 1]
    best_test_metrics = compute_metrics(y_test.to_numpy(), best_test_probs, threshold=best_threshold)
    plot_confusion_matrix(
        best_test_metrics,
        title=f"{best_name} (Threshold = {best_threshold})",
        save_path=RESULTS_DIR / "confusion_matrix.png",
    )

    # 8. Feature Importance for Best Model
    clf = best_pipe.named_steps["classifier"]
    feature_names = list(X.columns)
    importance_df = pd.DataFrame({"feature": feature_names, "importance": np.zeros(len(feature_names))})

    if hasattr(clf, "feature_importances_"):
        importance_df["importance"] = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        importance_df["importance"] = np.abs(clf.coef_[0])
    else:
        # Fallback rank
        importance_df["importance"] = np.linspace(1.0, 0.1, len(feature_names))

    importance_df = importance_df.sort_values("importance", ascending=False).reset_index(drop=True)
    importance_df.to_csv(RESULTS_DIR / "feature_importance.csv", index=False)
    plot_feature_importance(importance_df, top_n=10, save_path=RESULTS_DIR / "feature_importance.png")

    # 9. Save Artifacts
    artifact = {
        "pipeline": best_pipe,
        "model": best_pipe,
        "threshold": best_threshold,
        "best_threshold": best_threshold,
        "model_name": best_name,
        "feature_names": feature_names,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data_source": source,
    }
    joblib.dump(artifact, ARTIFACT_PATH)

    metrics_payload = {
        "best_model": best_name,
        "threshold": best_threshold,
        "data_source": source,
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
        "test_metrics": best_test_metrics,
        "comparison": comparison_rows,
    }
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics_payload, indent=2))

    total_time = time.time() - start_total
    if verbose:
        print("\n" + "=" * 65)
        print("✅ Training Pipeline Complete!")
        print(f"Saved best model -> {ARTIFACT_PATH}")
        print(f"Saved metrics & charts -> {RESULTS_DIR}")
        print(f"Total time elapsed: {total_time:.1f}s")
        print("=" * 65)

    return {
        "best_model": best_name,
        "threshold": best_threshold,
        "test_metrics": best_test_metrics,
        "model_comparison": comp_df,
    }
