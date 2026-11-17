<div align="center">

# 💳 Credit Card Fraud Detection using Machine Learning

**An end-to-end, production-ready ML system that flags fraudulent card transactions in highly imbalanced streams.**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Production%20API-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-ff4b4b.svg)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)
[![Tests](https://img.shields.io/badge/Tests-5%20Passed-brightgreen.svg)](tests/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

*Final-Year B.Tech Computer Science & Engineering Capstone Project*  
**IILM University**

**Student Investigators:**  
**Kritika Pathak** (Roll No: CS-23411554) & **Nida Fatima** (Roll No: CS-2341757)  
**Project Guide:** **Ms. Shiwangi Shukla**

</div>

---

## 📌 Project Status

| Component | Status | Description |
|---|---|---|
| **Data Cleaning & Engineering** | ✅ Complete | Duplicate removal, `Hour` and `Log_Amount` features, `RobustScaler` (leakage-free) |
| **Model Training & Comparison** | ✅ Complete | 4 paradigms: Logistic Regression, Random Forest, HistGB, Isolation Forest |
| **Evaluation Suite** | ✅ Complete | PR-AUC, ROC-AUC, Recall, Precision, F1, MCC, Confusion Matrix, Feature Importance |
| **Validation Threshold Tuning** | ✅ Complete | F1-optimal operating threshold computed strictly on validation split |
| **Interactive Web Application** | ✅ Complete | Modern responsive UI (`index.html`) served automatically via FastAPI root `/` |
| **FastAPI REST Service** | ✅ Complete | Real-time `/predict`, batch `/predict/batch`, `/health`, and `/docs` Swagger |
| **Streamlit Analytics Dashboard** | ✅ Complete | 6 interactive tabs: Explorer, Trainer, Performance, Single Checker, Batch CSV |
| **Automated Test Suite** | ✅ Complete | 100% passing tests via `pytest` covering schema, preprocessing, and inference |
| **Containerization & Deployment** | ✅ Complete | Multi-stage `Dockerfile`, `docker-compose.yml`, and `Procfile` ready for cloud |
| **Academic Documentation** | ✅ Complete | Formal Capstone Report & **25 Viva Voce Questions & Answers** in `docs/` |

---

## 📖 Overview

Credit-card fraud is exceedingly rare (**~0.17%** of transactions in benchmark datasets), yet financially devastating. Under this extreme class imbalance, a naive model that always predicts "Legitimate" scores 99.8% accuracy while failing to detect a single crime. 

This project solves this challenge through a rigorous, statistically grounded machine learning system:
- **Zero Data Leakage:** Preprocessing and robust scaling are fitted strictly on the training partition.
- **Rare-Event Metric Optimization:** Models are optimized for **Precision-Recall AUC (PR-AUC)** and Matthews Correlation Coefficient (MCC) rather than misleading accuracy.
- **Validation Threshold Tuning:** Decision thresholds are tuned on a dedicated validation split to maximize detection recall while keeping false alarms to an absolute minimum.
- **Dual Interface Architecture:** Provides both an enterprise-grade REST API with a modern web dashboard and a Streamlit data science workbench.

---

## 🗂️ Dataset

Benchmark dataset: [Kaggle - Credit Card Fraud Detection (ULB Machine Learning Group)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

| Property | Value |
|---|---|
| **Total Transactions** | 284,807 (European cardholder transactions over two days) |
| **Fraud Cases** | 492 (0.172% minority class) |
| **Features** | `Time`, `V1`–`V28` (PCA-anonymized for GDPR compliance), `Amount` |
| **Target Label** | `Class` ($0 = \text{Legitimate}, 1 = \text{Fraudulent}$) |

*Note: If `data/creditcard.csv` is not present, the system automatically generates and loads a high-fidelity synthetic demo dataset with matching schema and statistical characteristics so the entire project runs out-of-the-box.*

---

## 📊 Model Comparison & Results

Evaluated across four distinct machine learning paradigms (64% train / 16% validation / 20% test holdout):

| Model | Val PR-AUC | Test Precision | Test Recall | Test F1 | Test PR-AUC | Test ROC-AUC | Test MCC |
|---|---|---|---|---|---|---|---|
| **Random Forest (Champion)** | **0.842** | **0.945** | **0.726** | **0.821** | **0.815** | **0.941** | **0.828** |
| **HistGradientBoosting** | 0.735 | 0.893 | 0.705 | 0.788 | 0.719 | 0.929 | 0.793 |
| **Logistic Regression** | 0.701 | 0.859 | 0.768 | 0.811 | 0.683 | 0.965 | 0.812 |
| **Isolation Forest** | 0.142 | 0.227 | 0.179 | 0.200 | 0.114 | 0.935 | 0.200 |

**Champion Model Performance:** Catches ~73% of all fraud instances while generating only 4 false alarms across 56,651 legitimate transactions (false alarm rate < 0.01%).  
**Most Influential Features:** `V14`, `V4`, `V3`, `V11`, and `V20`.

---

## 🚀 Quick Start Guide

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/kritikapathak04/Credit-Card-Fraud-Detection-Using-ML.git
cd Credit-Card-Fraud-Detection-Using-ML

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate          # On Windows
# source venv/bin/activate     # On Linux / macOS

# Install dependencies
pip install -r requirements.txt
```

---

### 2. Run the Applications

#### Option A: Unified Web App & FastAPI Service (Recommended)
Starts the high-performance backend and serves the interactive dashboard directly:
```bash
uvicorn api:app --reload --port 8000
```
- 🌐 **Web Dashboard:** Open [http://localhost:8000](http://localhost:8000) in your browser.
- 📑 **Interactive Swagger API Docs:** Open [http://localhost:8000/docs](http://localhost:8000/docs).

#### Option B: Streamlit Analytical Dashboard
Launch the multi-page exploratory and retraining workspace:
```bash
streamlit run app.py
```
- Opens [http://localhost:8501](http://localhost:8501) with live EDA charts, batch CSV prediction, and retraining controls.

#### Option C: Docker 1-Command Deployment
```bash
docker-compose up --build
```

---

### 3. Model Training & Testing

```bash
# Retrain all models with fast mode
python train.py --fast

# Retrain with full hyperparameter search
python train.py

# Run automated test suite
python -m pytest -v
```

---

## 📂 Project Architecture

```
Credit-Card-Fraud-Detection-Using-ML/
├── api.py                    # Unified FastAPI REST service (serves API + web dashboard)
├── app.py                    # Streamlit analytical dashboard
├── index.html                # Responsive web application frontend
├── train.py                  # CLI training entry point
├── generate_sample_data.py   # Synthetic demo dataset generator
├── Dockerfile                # Production multi-stage Docker container
├── docker-compose.yml        # Docker composition file
├── Procfile                  # Cloud deployment configuration (Render / Railway)
├── requirements.txt          # Unified project dependencies (ML, API, Dashboard)
│
├── src/                      # Modular Python source package
│   ├── config.py             # Global constants, paths, and hyperparameters
│   ├── data_loader.py        # Kaggle loader & synthetic fallback generator
│   ├── preprocessing.py      # Deduplication, Hour/Log_Amount, RobustScaler
│   ├── models.py             # 4 Candidate models (RF, HistGB, LogReg, IsoForest)
│   ├── evaluate.py           # Evaluation metrics, F1 threshold tuning, plot generation
│   ├── training.py           # Stratified 64/16/20 training orchestrator
│   └── predict.py            # Inference, risk categorization, and batch scoring
│
├── tests/
│   └── test_pipeline.py      # Pytest automated test suite
│
├── docs/
│   └── PROJECT_REPORT.md     # Formal Capstone Report & 25 Viva Voce Q&As
│
├── data/
│   └── sample_creditcard.csv # Generated demo dataset
│
├── models/
│   └── best_model.joblib     # Serialized champion pipeline + tuned threshold
│
└── results/                  # Generated evaluation artifacts
    ├── metrics.json          # Numerical test metrics
    ├── model_comparison.csv  # Comparative metrics table
    ├── roc_curves.png        # Comparative ROC curves
    ├── pr_curves.png         # Precision-Recall curves
    ├── confusion_matrix.png  # Annotated confusion matrix
    ├── feature_importance.png# Top influential features bar chart
    └── class_distribution.png# Log-scale class balance chart
```

---

## 🎓 College Examination & Viva Voce Preparation

For your final evaluation and viva voce defense, a comprehensive academic project report containing detailed theory, mathematical derivations, and **25 common viva questions and model answers** is available in:

👉 **[docs/PROJECT_REPORT.md](docs/PROJECT_REPORT.md)**

Key viva topics covered:
1. Handling extreme class imbalance without data leakage.
2. Why PR-AUC outperforms ROC-AUC in rare-event domains.
3. Decision threshold optimization using the validation set.
4. Mathematical rationale for log-transforming transaction amounts and using `RobustScaler`.
5. Production latency, concept drift, and scaling to thousands of transactions per second.

---

## 👥 Authors & Acknowledgments

- **Kritika Pathak** (CS-23411554) — *Department of CSE, IILM University*
- **Nida Fatima** (CS-2341757) — *Department of CSE, IILM University*
- **Project Guide:** **Ms. Shiwangi Shukla** — *Assistant Professor, Department of CSE, IILM University*
- **Dataset Acknowledgments:** Worldline and the Machine Learning Group (MLG) of Université Libre de Bruxelles (ULB).

---

## 📜 License
This project is open-source and licensed under the [MIT License](LICENSE).
