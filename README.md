# 💳 Credit Card Fraud Detection using Machine Learning

Final-year (Semester 7) capstone project. An end-to-end ML system that detects fraudulent credit-card
transactions: data cleaning → feature engineering → imbalance-aware training of 4 models → threshold tuning →
evaluation → interactive Streamlit dashboard (single check + CSV batch scoring).

Inspired by the open-source projects
[ashish-kharde1/credit-card-fraud-detection](https://github.com/ashish-kharde1/credit-card-fraud-detection) and
[mohammadhashim135/Credit-Card-Fraud-Detection](https://github.com/mohammadhashim135/Credit-Card-Fraud-Detection),
rebuilt as an original codebase with a proper evaluation methodology.

## Dataset
[Kaggle - Credit Card Fraud Detection (ULB Machine Learning Group)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud):
284,807 transactions by European cardholders over two days, only 492 (0.172%) fraudulent.
Features: `Time`, `V1-V28` (PCA-anonymised), `Amount`, label `Class` (1 = fraud).

**Setup:** download `creditcard.csv` and put it at `data/creditcard.csv`
(manually, or `kaggle datasets download -d mlg-ulb/creditcardfraud -p data --unzip`).
The real file is already included in `data/`. If it is ever missing, the project automatically uses a **synthetic demo dataset** with the same schema so
everything still runs - but its numbers are NOT real results and the app says so.

## Quick start
```bash
python -m venv venv
venv\Scripts\activate            # Windows   (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt

python train.py                  # trains + evaluates all models, saves models/ and results/
python train.py --fast           # quicker run with smaller models
streamlit run app.py             # open the dashboard
pytest -q                        # run tests
```
You can also train from inside the app (sidebar → **Train Model**).
On the full 284k-row dataset, Random Forest takes a few minutes; use `--fast` while developing.

## Project structure
```
├── app.py                  Streamlit dashboard (6 pages)
├── train.py                CLI entry point for training
├── generate_sample_data.py Creates the synthetic demo dataset
├── requirements.txt
├── src/
│   ├── config.py           paths, column names, constants
│   ├── data_loader.py      loads real CSV or synthetic fallback; schema validation
│   ├── preprocessing.py    cleaning, feature engineering (Hour, Log_Amount), RobustScaler pipeline
│   ├── models.py           Logistic Regression, Random Forest, HistGradientBoosting, Isolation Forest
│   ├── evaluate.py         metrics, F1-optimal threshold, plots
│   ├── training.py         split → fit → tune → evaluate → save
│   └── predict.py          inference helpers + risk levels
├── tests/test_pipeline.py
├── docs/PROJECT_REPORT.md  report skeleton + viva questions
├── data/    models/    results/    (generated)
```

## Methodology (what makes it more than "fit a classifier")
| Step | Choice | Why |
|---|---|---|
| Split | 64% train / 16% validation / 20% test, stratified | Model selection and threshold tuning never touch the test set |
| Cleaning | drop duplicates / missing | Kaggle file contains ~1,000 duplicate rows |
| Features | `Hour` from `Time`, `Log_Amount`, RobustScaler on Amount/Time | Amount is heavily skewed |
| Imbalance | class weights (`balanced`) | Avoids leakage that comes from oversampling before splitting |
| Models | LR, Random Forest, HistGradientBoosting, Isolation Forest | linear baseline, bagging, boosting, unsupervised anomaly detection |
| Metrics | PR-AUC (main), recall, precision, F1, MCC, ROC-AUC | Accuracy is ~99.8% even for a model that flags nothing |
| Threshold | maximise F1 on validation set | 0.5 is rarely optimal for rare events |
| Explainability | permutation importance on validation data | shows which features drive detection |


## Results on the real Kaggle dataset
Run with `python train.py --fast` (smaller Random Forest: 60 trees). 283,726 rows after removing 1,081 duplicates
(473 frauds, 0.167%). Hold-out test set: 56,746 transactions containing **95 frauds**.
Best model chosen on validation PR-AUC: **Random Forest**, decision threshold 0.416.

| Model | Precision | Recall | F1 | PR-AUC | ROC-AUC | MCC |
|---|---|---|---|---|---|---|
| Random Forest | 0.945 | 0.726 | 0.821 | 0.815 | 0.941 | 0.828 |
| Gradient Boosting (HistGB) | 0.893 | 0.705 | 0.788 | 0.719 | 0.929 | 0.793 |
| Logistic Regression | 0.859 | 0.768 | 0.811 | 0.683 | 0.965 | 0.812 |
| Isolation Forest (unsupervised) | 0.227 | 0.179 | 0.200 | 0.114 | 0.935 | 0.200 |

Random Forest on the test set: 69 of 95 frauds caught, 26 missed, only 4 false alarms among 56,651 legitimate transactions.
Most important features: V14, V4, V3, V11, V20.

**Caveat:** with only 95 frauds in the test set, each metric moves by several points depending on the random seed
or split, so report these as approximate and mention this in your discussion. Running without `--fast` (150 trees)
may change the numbers slightly; re-run and quote your own final `results/metrics.json`.

## Outputs
`results/`: `metrics.json`, `model_comparison.csv`, `roc_curves.png`, `pr_curves.png`,
`confusion_matrix.png`, `feature_importance.png/csv`, `class_distribution.png`.
`models/best_model.joblib`: full pipeline + tuned threshold.

## Limitations & future work
- Features are anonymised, so decisions cannot be explained in business terms (e.g. merchant, location).
- Data covers two days; real fraud patterns drift over time → periodic retraining / concept-drift monitoring.
- Future scope: SMOTE/cost-sensitive ablation, XGBoost/LightGBM, autoencoder anomaly detection, SHAP,
  REST API (FastAPI) + Docker, streaming scoring with Kafka.

## License
MIT - for educational use.
