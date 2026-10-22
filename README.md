<div align="center">

# 💳 Credit Card Fraud Detection using Machine Learning

**An end-to-end ML system that flags fraudulent card transactions in a highly imbalanced dataset.**

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-dashboard-ff4b4b)
![Backend](https://img.shields.io/badge/ML%20pipeline-complete-brightgreen)
![Frontend](https://img.shields.io/badge/Frontend-in%20progress-yellow)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

*Final-year (Semester 7) capstone project*

</div>

---

## 📌 Project Status

| Component | Status | Notes |
|---|---|---|
| Data loading, cleaning, feature engineering | ✅ Complete | Duplicate removal, `Hour` and `Log_Amount` features, robust scaling |
| Model training and comparison (4 models) | ✅ Complete | Validated on the real Kaggle dataset |
| Evaluation (PR-AUC, recall, precision, F1, MCC, ROC-AUC) | ✅ Complete | Plots and tables saved to `results/` |
| Threshold tuning and model selection | ✅ Complete | Done on a separate validation set |
| Automated tests | ✅ Complete | `pytest` covers data schema, training and prediction |
| **Streamlit dashboard (frontend)** | 🚧 **In progress** | Pages are drafted but still being tested, polished and extended |
| Deployment (Docker / API) | 📝 Planned | See [Roadmap](#-roadmap) |

> **Heads-up:** the machine-learning pipeline is finished and can be run entirely from the command line.
> The web interface is **still under development**: expect rough edges, layout changes and missing features.

---

## 📖 Overview

Credit-card fraud is rare (**0.17%** of transactions in this dataset) but costly, so a model that always answers
"legitimate" is ~99.8% accurate and completely useless. This project is built around that problem:

- compares a linear baseline, bagging, boosting and an unsupervised anomaly detector,
- judges them with metrics suited to rare events (PR-AUC, recall, precision) instead of accuracy,
- keeps the test set untouched until the final report (no data leakage),
- tunes the decision threshold instead of using the default 0.5.

## 🗂️ Dataset

[Kaggle - Credit Card Fraud Detection (ULB Machine Learning Group)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

| | |
|---|---|
| Transactions | 284,807 (European cardholders, two days) |
| Frauds | 492 (0.172%) |
| Features | `Time`, `V1`-`V28` (PCA-anonymised), `Amount` |
| Label | `Class` (1 = fraud, 0 = legitimate) |

The file is expected at `data/creditcard.csv` (included in this archive). To download it yourself:
`kaggle datasets download -d mlg-ulb/creditcardfraud -p data --unzip`.
If the file is missing, a **synthetic demo dataset** with the same columns is generated automatically so the code
still runs. Its numbers are *not* real results, and the app labels it clearly.

## 📊 Results (real dataset)

Run with `python train.py --fast` (60-tree Random Forest). After removing 1,081 duplicates: 283,726 rows, 473 frauds.
Hold-out test set: 56,746 transactions containing **95 frauds**. Model chosen on validation PR-AUC.

| Model | Precision | Recall | F1 | PR-AUC | ROC-AUC | MCC |
|---|---|---|---|---|---|---|
| **Random Forest** (selected, threshold 0.416) | 0.945 | 0.726 | 0.821 | 0.815 | 0.941 | 0.828 |
| Gradient Boosting (HistGB) | 0.893 | 0.705 | 0.788 | 0.719 | 0.929 | 0.793 |
| Logistic Regression | 0.859 | 0.768 | 0.811 | 0.683 | 0.965 | 0.812 |
| Isolation Forest (unsupervised) | 0.227 | 0.179 | 0.200 | 0.114 | 0.935 | 0.200 |

**Selected model on the test set:** 69 of 95 frauds caught, 26 missed, 4 false alarms among 56,651 legitimate
transactions. Most influential features: `V14`, `V4`, `V3`, `V11`, `V20`.

> ⚠️ With only 95 frauds in the test set, each metric can move by several points with a different split or seed.
> Treat these as approximate. A full run (`python train.py`, 150 trees) may change the numbers slightly.

## 🧠 Methodology

| Step | Choice | Why |
|---|---|---|
| Split | 64% train / 16% validation / 20% test, stratified | Model selection and threshold tuning never touch the test set |
| Cleaning | Drop duplicates and missing values | The Kaggle file contains ~1,000 duplicate rows |
| Features | `Hour` from `Time`, `Log_Amount`, `RobustScaler` on Amount/Time | `Amount` is heavily skewed; scaler is fitted on training data only |
| Imbalance | Class weights (`balanced`) | Avoids the leakage that comes from oversampling before splitting |
| Models | Logistic Regression, Random Forest, HistGradientBoosting, Isolation Forest | Linear baseline, bagging, boosting, unsupervised anomaly detection |
| Metrics | PR-AUC (main), recall, precision, F1, MCC, ROC-AUC | Accuracy is misleading on imbalanced data |
| Threshold | Maximise F1 on the validation set | 0.5 is rarely optimal for rare events |
| Explainability | Permutation importance on validation data | Shows which features drive detection |

## 🚀 Getting Started

**Prerequisites:** Python 3.9+ and `pip`.

```bash
# 1. create and activate a virtual environment
python -m venv venv
venv\Scripts\activate              # Windows   (macOS/Linux: source venv/bin/activate)

# 2. install dependencies
pip install -r requirements.txt

# 3. train and evaluate all models (saves to models/ and results/)
python train.py                    # full run, takes a few minutes
python train.py --fast             # quicker run with smaller models

# 4. (optional) run the tests
pytest -q

# 5. (work in progress) launch the dashboard
streamlit run app.py
```

You can run steps 1-4 without ever opening the dashboard; everything the dashboard shows is produced by `train.py`.

## 🖥️ Dashboard (work in progress)

| Page | Purpose | Status |
|---|---|---|
| Overview | Dataset summary and project explanation | 🚧 Drafted |
| Data Explorer | Class balance, amount distribution, correlations, fraud by hour | 🚧 Drafted |
| Train Model | Train all models from the browser | 🚧 Drafted |
| Model Performance | Metrics, comparison table, ROC / PR curves, confusion matrix | 🚧 Drafted |
| Check a Transaction | Score one transaction (load a sample, edit values) | 🚧 Drafted |
| Batch Detection | Upload a CSV, get predictions and download results | 🚧 Drafted |

Known gaps: not yet fully tested across browsers and Streamlit versions, UI styling is basic, and there are no
screenshots yet.

## 📂 Project Structure

```
├── app.py                   Streamlit dashboard (in progress)
├── train.py                 CLI entry point for training
├── generate_sample_data.py  Creates the synthetic demo dataset
├── requirements.txt
├── LICENSE
├── src/
│   ├── config.py            paths, column names, constants
│   ├── data_loader.py       loads real CSV or synthetic fallback; schema validation
│   ├── preprocessing.py     cleaning, feature engineering, scaling pipeline
│   ├── models.py            the four candidate models
│   ├── evaluate.py          metrics, F1-optimal threshold, plots
│   ├── training.py          split -> fit -> tune -> evaluate -> save
│   └── predict.py           inference helpers and risk levels
├── tests/test_pipeline.py
├── docs/PROJECT_REPORT.md   report skeleton and viva questions
├── data/                    creditcard.csv
├── models/                  best_model.joblib (generated)
└── results/                 metrics, tables and plots (generated)
```

**Outputs:** `results/` holds `metrics.json`, `model_comparison.csv`, `roc_curves.png`, `pr_curves.png`,
`confusion_matrix.png`, `feature_importance.png/csv` and `class_distribution.png`. `models/best_model.joblib`
contains the full pipeline plus the tuned threshold.

## 🛣️ Roadmap

- [x] Leakage-free preprocessing and 4-model comparison
- [x] Threshold tuning, permutation importance, unit tests
- [ ] Finish and test the Streamlit dashboard, add screenshots
- [ ] Multi-seed evaluation (mean ± std) to account for the small number of test frauds
- [ ] Resampling ablation (SMOTE / undersampling vs. class weights)
- [ ] XGBoost / LightGBM and SHAP explanations
- [ ] REST API (FastAPI) and Docker image
- [ ] Concept-drift monitoring and scheduled retraining

## ⚠️ Limitations

- Features are PCA-anonymised, so predictions cannot be explained in business terms (merchant, location, etc.).
- The data covers only two days; real fraud patterns drift over time.
- Very few fraud cases in the test set, so results have high variance.
- Built for learning and demonstration, not for production fraud screening.

## 🙏 Acknowledgements

- Dataset: Worldline and the [ULB Machine Learning Group](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
- Inspiration: [ashish-kharde1/credit-card-fraud-detection](https://github.com/ashish-kharde1/credit-card-fraud-detection)
  and [mohammadhashim135/Credit-Card-Fraud-Detection](https://github.com/mohammadhashim135/Credit-Card-Fraud-Detection).
  This codebase was written independently.

## 📜 License

Released under the MIT License (see `LICENSE`). For educational use.
