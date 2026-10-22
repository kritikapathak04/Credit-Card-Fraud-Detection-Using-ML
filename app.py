"""Streamlit dashboard for the Credit Card Fraud Detection capstone project.
Run:  streamlit run app.py
"""
import json

import pandas as pd
import streamlit as st

from src.config import ARTIFACT_PATH, RAW_FEATURES, RESULTS_DIR, TARGET, V_COLS
from src.data_loader import load_data
from src.predict import load_artifact, predict_df, risk_level
from src.training import run_training

st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="💳", layout="wide")


# ------------------------------------------------------------------ cached loaders
@st.cache_data(show_spinner="Loading dataset...")
def get_data():
    return load_data()


@st.cache_resource(show_spinner="Loading model...")
def get_artifact(mtime: float):
    return load_artifact()


def model_ready() -> bool:
    return ARTIFACT_PATH.exists()


def artifact():
    return get_artifact(ARTIFACT_PATH.stat().st_mtime)


def need_model():
    st.warning("No trained model found. Go to **Train Model** in the sidebar (or run `python train.py`).")
    st.stop()


# ------------------------------------------------------------------ pages
def page_overview():
    st.title("💳 Credit Card Fraud Detection")
    st.caption("Machine-learning system to flag fraudulent card transactions - Final-year capstone project")
    c1, c2, c3 = st.columns(3)
    df, source = get_data()
    c1.metric("Transactions", f"{len(df):,}")
    c2.metric("Fraud cases", f"{int(df[TARGET].sum()):,}")
    c3.metric("Fraud rate", f"{100 * df[TARGET].mean():.3f}%")
    st.info(f"**Data source:** {source}")
    if "synthetic" in source or "custom" in source:
        st.warning("This is NOT the official Kaggle dataset, so reported numbers are not real-world results. "
                   "Download `creditcard.csv` from Kaggle into the `data/` folder and re-train.")
    st.subheader("How it works")
    st.markdown(
        """
1. **Data** - 284,807 European card transactions (2 days); 28 anonymised PCA features `V1-V28`, `Time`, `Amount`, label `Class`.
2. **Cleaning & features** - duplicates removed, `Hour` and `Log_Amount` engineered, `Amount`/`Time` robust-scaled (fitted on training data only).
3. **Imbalance handling** - class weights (fraud is ~0.17% of transactions), so no synthetic rows leak into evaluation.
4. **Models compared** - Logistic Regression, Random Forest, Gradient Boosting, Isolation Forest (unsupervised).
5. **Evaluation** - Precision-Recall AUC (main metric), recall, precision, F1, MCC, ROC-AUC. Accuracy is misleading here.
6. **Threshold tuning** - decision threshold chosen on a validation set to maximise F1; test set stays untouched.
        """
    )


def page_explorer():
    st.title("📊 Data Explorer")
    df, source = get_data()
    st.caption(f"Source: {source}")
    st.dataframe(df.head(100))
    import matplotlib.pyplot as plt
    import numpy as np

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(5, 4))
        counts = df[TARGET].value_counts().sort_index()
        ax.bar(["Legitimate", "Fraud"], counts.values, color=["#4C9F70", "#D1495B"])
        ax.set_yscale("log"); ax.set_title("Class distribution (log scale)")
        st.pyplot(fig); plt.close(fig)
    with c2:
        fig, ax = plt.subplots(figsize=(5, 4))
        for cls, color, lab in [(0, "#4C9F70", "Legitimate"), (1, "#D1495B", "Fraud")]:
            ax.hist(np.log1p(df.loc[df[TARGET] == cls, "Amount"]), bins=40, alpha=0.6, color=color,
                    label=lab, density=True)
        ax.set_xlabel("log(1 + Amount)"); ax.set_title("Amount distribution"); ax.legend()
        st.pyplot(fig); plt.close(fig)

    st.subheader("Features most correlated with fraud")
    corr = df[V_COLS + ["Amount", "Time"]].corrwith(df[TARGET]).sort_values(key=abs, ascending=False).head(12)
    fig, ax = plt.subplots(figsize=(7, 4))
    corr.iloc[::-1].plot.barh(ax=ax, color="#3E7CB1")
    ax.set_xlabel("Pearson correlation with Class")
    st.pyplot(fig); plt.close(fig)

    st.subheader("Fraud by hour of day")
    hours = ((df["Time"] // 3600) % 24).astype(int)
    st.bar_chart(df.groupby(hours)[TARGET].mean().rename("fraud rate"))


def page_train():
    st.title("🛠️ Train Model")
    df, source = get_data()
    st.write(f"Data source: **{source}** - {len(df):,} rows")
    fast = st.checkbox("Fast mode (smaller models, quicker)", value=len(df) > 100_000)
    if st.button("Train all models", type="primary"):
        with st.spinner("Training... Random Forest on the full dataset can take a few minutes."):
            summary = run_training(df, fast=fast, source=source, verbose=False)
        get_artifact.clear()
        st.success(f"Done. Best model: {summary['best_model']}")
        st.rerun()
    if model_ready():
        a = artifact()
        st.write(f"Current model: **{a['model_name']}** (trained {a['trained_at']}, data: {a['data_source']})")


def page_performance():
    st.title("📈 Model Performance")
    if not model_ready():
        need_model()
    m = json.loads((RESULTS_DIR / "metrics.json").read_text())
    t = m["test_metrics"]
    st.write(f"Best model: **{m['best_model']}** | threshold **{m['threshold']:.3f}** | data: {m['data_source']}")
    cols = st.columns(5)
    for col, (lab, key) in zip(cols, [("Precision", "precision"), ("Recall", "recall"), ("F1", "f1"),
                                      ("PR-AUC", "pr_auc"), ("ROC-AUC", "roc_auc")]):
        col.metric(lab, f"{t[key]:.3f}")
    st.caption(f"Test set: {m['n_test']:,} transactions. Caught {t['tp']} frauds, missed {t['fn']}, "
               f"{t['fp']} false alarms.")
    st.subheader("Model comparison (test set)")
    comp = pd.read_csv(RESULTS_DIR / "model_comparison.csv")
    show = comp[["model", "val_pr_auc", "test_precision", "test_recall", "test_f1", "test_pr_auc",
                 "test_roc_auc", "test_mcc", "train_seconds"]]
    st.dataframe(show.round(4))
    c1, c2 = st.columns(2)
    c1.image(str(RESULTS_DIR / "roc_curves.png"))
    c2.image(str(RESULTS_DIR / "pr_curves.png"))
    c3, c4 = st.columns(2)
    c3.image(str(RESULTS_DIR / "confusion_matrix.png"))
    c4.image(str(RESULTS_DIR / "feature_importance.png"))


def _init_inputs():
    for c in RAW_FEATURES:
        st.session_state.setdefault(f"in_{c}", 0.0)


def _load_sample(cls: int):
    df, _ = get_data()
    row = df[df[TARGET] == cls].sample(1).iloc[0]
    for c in RAW_FEATURES:
        st.session_state[f"in_{c}"] = float(row[c])


def page_single():
    st.title("🔍 Check a Transaction")
    if not model_ready():
        need_model()
    a = artifact()
    _init_inputs()
    st.caption("Features V1-V28 are anonymised PCA components, so load a real sample to try the model, then edit values.")
    b1, b2, _ = st.columns([1, 1, 4])
    b1.button("Load legitimate sample", on_click=_load_sample, args=(0,))
    b2.button("Load fraud sample", on_click=_load_sample, args=(1,))
    c1, c2 = st.columns(2)
    c1.number_input("Time (seconds since first transaction)", key="in_Time", format="%.0f")
    c2.number_input("Amount", key="in_Amount", format="%.2f", min_value=0.0)
    with st.expander("PCA features V1 - V28", expanded=False):
        cols = st.columns(4)
        for i, c in enumerate(V_COLS):
            cols[i % 4].number_input(c, key=f"in_{c}", format="%.4f")
    if st.button("Check transaction", type="primary"):
        row = pd.DataFrame([{c: st.session_state[f"in_{c}"] for c in RAW_FEATURES}])
        out = predict_df(a, row).iloc[0]
        p = float(out["fraud_probability"])
        level = risk_level(p, a["threshold"])
        if out["prediction"] == 1:
            st.error(f"🚨 FLAGGED AS FRAUD - probability {p:.1%} (risk: {level})")
        else:
            st.success(f"✅ Looks legitimate - fraud probability {p:.1%} (risk: {level})")
        st.progress(min(max(p, 0.0), 1.0))
        st.caption(f"Model: {a['model_name']} | decision threshold: {a['threshold']:.3f}")


def page_batch():
    st.title("📂 Batch Detection (CSV upload)")
    if not model_ready():
        need_model()
    a = artifact()
    st.write("Upload a CSV with columns: `Time, V1..V28, Amount` (a `Class` column is optional).")
    df, _ = get_data()
    st.download_button("Download a sample CSV", df.sample(min(200, len(df)), random_state=1)
                       .drop(columns=[TARGET]).to_csv(index=False), "sample_transactions.csv", "text/csv")
    up = st.file_uploader("Upload CSV", type="csv")
    if up is None:
        return
    try:
        data = pd.read_csv(up)
        res = predict_df(a, data)
    except Exception as e:  # schema problems etc.
        st.error(f"Could not process file: {e}")
        return
    n_fraud = int(res["prediction"].sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Transactions", f"{len(res):,}")
    c2.metric("Flagged as fraud", f"{n_fraud:,}")
    c3.metric("Flag rate", f"{100 * n_fraud / len(res):.2f}%")
    if TARGET in res.columns:
        tp = int(((res["prediction"] == 1) & (res[TARGET] == 1)).sum())
        st.caption(f"Labels found: {tp} of {int(res[TARGET].sum())} actual frauds detected.")
    st.subheader("Top suspicious transactions")
    st.dataframe(res.sort_values("fraud_probability", ascending=False).head(50))
    st.download_button("Download all results (CSV)", res.to_csv(index=False), "fraud_predictions.csv", "text/csv")


# ------------------------------------------------------------------ navigation
PAGES = {
    "Overview": page_overview,
    "Data Explorer": page_explorer,
    "Train Model": page_train,
    "Model Performance": page_performance,
    "Check a Transaction": page_single,
    "Batch Detection": page_batch,
}
st.sidebar.title("Navigation")
choice = st.sidebar.radio("Go to", list(PAGES))
st.sidebar.markdown("---")
st.sidebar.caption("Model status: " + ("✅ trained" if model_ready() else "❌ not trained"))
PAGES[choice]()
