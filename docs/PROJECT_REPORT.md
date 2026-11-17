# 💳 Credit Card Fraud Detection using Machine Learning
## Final Year Capstone Project Report & Viva Voce Guide

**Academic Year:** 2026–2027  
**Department:** Computer Science & Engineering  
**Institution:** IILM University  
**Project Guide:** Ms. Shiwangi Shukla  
**Authors:**
- **Kritika Pathak** (Roll No: CS-23411554)
- **Nida Fatima** (Roll No: CS-2341757)

---

## 📑 Executive Summary / Abstract
Credit card fraud represents one of the most critical threats facing modern digital banking and e-commerce ecosystems, leading to tens of billions of dollars in financial losses annually. Designing an automated, machine-learning-based fraud detection system is challenging primarily due to the **extreme class imbalance** (where fraudulent transactions account for only ~0.17% of total volume) and the severe business cost asymmetry (missing a fraud is far more catastrophic than asking a customer for secondary authentication).

This project implements an end-to-end, production-ready machine learning system that:
1. Employs a strict **data-leakage-free preprocessing pipeline** using `RobustScaler` and domain feature engineering (`Hour` and `Log_Amount`).
2. Evaluates four distinct learning paradigms: linear baseline (**Logistic Regression**), bagging ensemble (**Random Forest**), gradient boosting (**HistGradientBoosting**), and unsupervised anomaly detection (**Isolation Forest**).
3. Utilizes rare-event metrics (**Precision-Recall AUC**, Recall, Precision, MCC) rather than misleading raw accuracy.
4. Tunes the decision threshold on a dedicated validation split to maximize the F1-score rather than assuming a naive 0.5 cutoff.
5. Deploys the trained champion model via both an interactive **Streamlit dashboard** and a high-performance **FastAPI REST web service with an integrated web interface and Docker containerization**.

---

## 1. Introduction & Problem Formulation
In standard classification benchmarks, classes are roughly balanced. However, in credit card transaction streams:
$$\text{Fraud Proportion } P(Y=1) \approx 0.172\%$$

If a trivial classifier predicts "Legitimate" ($Y=0$) for every single transaction, it achieves an apparent accuracy of **99.83%**, yet it catches **0% of frauds**, rendering accuracy a useless metric. 

### Core Objectives
1. Maximize fraud detection rate (**Recall**) while maintaining an acceptable false-alarm rate (**Precision**).
2. Compare linear, ensemble, and unsupervised anomaly detection paradigms.
3. Optimize decision thresholds on validation data without contaminating the test set.
4. Provide a dual-interface deployment architecture (API and Web Dashboard).

---

## 2. Dataset Description & Statistical Characteristics
The study utilizes the European Cardholders benchmark dataset (ULB Machine Learning Group / Worldline):
- **Total Transactions:** 284,807 across two consecutive days.
- **Fraudulent Transactions:** 492 (0.172%).
- **Features:** 30 numerical input features:
  - `Time`: Elapsed seconds since the first recorded transaction.
  - `Amount`: Transaction amount in Euros (heavily right-skewed).
  - `V1`–`V28`: Principal Component Analysis (PCA) transformed features to preserve customer confidentiality.
  - `Class`: Ground-truth label ($0 = \text{Legitimate}, 1 = \text{Fraud}$).

---

## 3. Methodology & System Architecture

```
Raw Transactions (creditcard.csv / Demo Fallback)
         │
         ▼
[ Data Cleaning: Deduplication & NaN Handling ]
         │
         ▼
[ Feature Engineering: Hour = (Time/3600)%24, Log_Amount = log1p(Amount) ]
         │
         ▼
[ Stratified Split: 64% Train │ 16% Validation │ 20% Test ]
         │
         ▼
[ RobustScaler Fitted Strictly on Training Split ]
         │
         ├─── Logistic Regression (class_weight='balanced')
         ├─── Random Forest (class_weight='balanced')
         ├─── HistGradientBoosting (class_weight='balanced')
         └─── Isolation Forest (unsupervised contamination=0.002)
         │
         ▼
[ Validation Evaluation: PR-AUC Comparison & F1 Threshold Optimization ]
         │
         ▼
[ Champion Model Selection: Random Forest at Tuned Threshold ]
         │
         ▼
[ Final Test Evaluation on Untouched 20% Holdout ]
         │
         ├─── Save Artifact: models/best_model.joblib
         ├─── Save Metrics: results/metrics.json & model_comparison.csv
         └─── Save Visualizations: results/*.png
         │
         ▼
[ Deployment Layer: FastAPI REST Service + Interactive Web Dashboard + Docker ]
```

### Feature Engineering & Preprocessing
1. **Hour Feature:** Derived from `Time` modulo 24 hours:
   $$\text{Hour} = \left\lfloor \frac{\text{Time}}{3600} \right\rfloor \pmod{24}$$
2. **Log-Transform on Amount:** Card transaction values span orders of magnitude:
   $$\text{Log\_Amount} = \ln(1 + \text{Amount})$$
3. **Robust Scaling:** Standard Z-score scaling is distorted by extreme transaction outliers. We apply `RobustScaler`, centering around the median and scaling to the Interquartile Range (IQR):
   $$x' = \frac{x - \text{median}(x)}{\text{IQR}(x)}$$

---

## 4. Evaluation Metrics for Imbalanced Datasets
Rather than overall accuracy, models are evaluated using:

1. **Precision:** $\frac{TP}{TP + FP}$ — The percentage of flagged transactions that were actually fraudulent.
2. **Recall (Sensitivity):** $\frac{TP}{TP + FN}$ — The percentage of total actual frauds successfully intercepted.
3. **F1-Score:** Harmonic mean balancing precision and recall:
   $$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$
4. **Precision-Recall AUC (PR-AUC):** Area under the Precision-Recall curve. Represents model quality across all possible operating thresholds without being inflated by true negatives.
5. **Matthews Correlation Coefficient (MCC):** Balanced correlation metric ranging from -1 to +1:
   $$\text{MCC} = \frac{TP \times TN - FP \times FN}{\sqrt{(TP + FP)(TP + FN)(TN + FP)(TN + FN)}}$$

---

## 5. Experimental Results & Model Comparison

| Model | Val PR-AUC | Test Precision | Test Recall | Test F1 | Test PR-AUC | Test ROC-AUC | Test MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Champion)** | **0.842** | **0.945** | **0.726** | **0.821** | **0.815** | **0.941** | **0.828** |
| **HistGradientBoosting** | 0.735 | 0.893 | 0.705 | 0.788 | 0.719 | 0.929 | 0.793 |
| **Logistic Regression** | 0.701 | 0.859 | 0.768 | 0.811 | 0.683 | 0.965 | 0.812 |
| **Isolation Forest** | 0.142 | 0.227 | 0.179 | 0.200 | 0.114 | 0.935 | 0.200 |

### Key Findings
- **Random Forest** achieved the highest PR-AUC and F1-score due to its ensemble bagging mechanism and resilience to non-linear feature interactions.
- **Threshold Tuning** on the validation set adjusted the decision threshold to optimal operating ranges rather than 0.5, maximizing caught frauds while keeping false alarms to a minimum.
- **Top Predictive Features:** Feature importance analysis identified `V14`, `V4`, `V3`, `V11`, and `V20` as the strongest statistical indicators of fraudulent activity.

---

## 6. System Deployment Architecture
The solution provides two deployment modalities:

1. **Production REST API & Web Dashboard (`api.py`):**
   - Built with **FastAPI** and **Uvicorn**.
   - Serves the responsive, modern client dashboard (`index.html`) directly at `http://localhost:8000/`.
   - Offers real-time scoring (`POST /predict`), batch processing (`POST /predict/batch`), health monitoring (`GET /health`), and Swagger documentation (`GET /docs`).
   - Packaged in a production **Dockerfile** with container healthchecks.
2. **Data Science Exploration Dashboard (`app.py`):**
   - Built with **Streamlit**.
   - Allows students, evaluators, and data scientists to explore class distributions, trigger model retraining from the browser, inspect PR/ROC curves, and test transactions interactively.

---

## 7. Viva Voce Examination Questions & Answers (25 Key Questions)

### Q1: Why is raw accuracy a misleading metric for credit card fraud detection?
**Answer:** In highly imbalanced datasets where fraud represents only ~0.17% of transactions, a naive model that predicts every transaction is legitimate achieves 99.83% accuracy while failing to detect a single fraudulent transaction. Metrics such as Precision-Recall AUC (PR-AUC), Recall, Precision, and F1-score are necessary to measure true detection capability.

### Q2: What is the difference between ROC-AUC and PR-AUC, and which is preferred here?
**Answer:** ROC-AUC plots True Positive Rate vs False Positive Rate. When the negative class is massive (99.83%), False Positive Rate ($\frac{FP}{FP+TN}$) remains tiny even when false alarms are numerous, giving an overly optimistic picture. PR-AUC plots Precision vs Recall and directly penalizes false positives, making it the preferred metric for rare-event detection.

### Q3: How did you prevent data leakage during preprocessing?
**Answer:** We performed the train-validation-test split **before** any feature scaling. The `RobustScaler` was fitted strictly on the training set, and the learned parameters (medians and IQRs) were then applied to transform validation and test sets.

### Q4: Why did you use `RobustScaler` instead of `StandardScaler` or `MinMaxScaler`?
**Answer:** Transaction amounts and times contain severe outliers. `StandardScaler` relies on the mean and standard deviation, which are skewed by outliers. `MinMaxScaler` compresses normal values into a narrow range when extreme outliers exist. `RobustScaler` uses the median and Interquartile Range (IQR), making it robust against extreme transaction values.

### Q5: Why was the decision threshold tuned on the validation set rather than the test set?
**Answer:** Tuning the threshold on the test set would introduce data leakage and optimistic bias. By finding the threshold that maximizes F1-score on the validation set, the test set remains completely untouched until final model verification.

### Q6: What does the feature `Hour` represent and why is it important?
**Answer:** Fraudulent transactions exhibit temporal clustering, often occurring during late-night or early-morning hours when cardholders are asleep. We derived `Hour = (Time / 3600) % 24` from elapsed seconds to capture this cyclical diurnal pattern.

### Q7: Why apply a log-transformation to transaction amount?
**Answer:** Transaction amounts are heavily right-skewed with high variance (most transactions are small, but a few are thousands of dollars). $\ln(1 + \text{Amount})$ normalizes the distribution and prevents large amounts from dominating linear and gradient calculations.

### Q8: What is class weighting and how does it compare to SMOTE?
**Answer:** Class weighting adjusts the loss function during training so that misclassifying a minority (fraud) instance incurs a significantly higher penalty proportional to $\frac{N}{2 \times N_{\text{fraud}}}$. Unlike SMOTE, which creates synthetic instances that can distort decision boundaries in high dimensions, class weighting introduces zero synthetic data leakage.

### Q9: Why was Random Forest selected as the champion model?
**Answer:** Random Forest achieved the highest validation PR-AUC (0.842) and test F1-score (0.821). Its ensemble of uncorrelated decision trees effectively handles non-linear boundaries and feature correlations without overfitting.

### Q10: How does Isolation Forest work and why was it included?
**Answer:** Isolation Forest is an unsupervised anomaly detector that recursively isolates observations using random feature splits. Anomalies require fewer splits to isolate (shorter tree path length). It was included as an unsupervised baseline to evaluate how well anomalies can be caught without labeled ground truth.

### Q11: Why are the features named V1 through V28?
**Answer:** For privacy and customer protection (under GDPR and financial secrecy laws), the bank applied Principal Component Analysis (PCA) to transform the original raw features (such as merchant category, user IP, device ID, location) into anonymized orthogonal components.

### Q12: What is the Matthews Correlation Coefficient (MCC)?
**Answer:** MCC is a contingency matrix metric that yields a score between -1 and +1, taking into account all four cells of the confusion matrix (TP, FP, TN, FN). It is regarded as one of the most reliable single-number metrics for imbalanced binary classification.

### Q13: What happens if an API request is submitted with missing PCA features?
**Answer:** The system imputes missing PCA features with 0.0 (the mean of standard PCA components) and returns a list of `missing_features_filled` in the API response, allowing inference to proceed gracefully while informing the client.

### Q14: How are risk tiers determined?
**Answer:** Transactions with predicted fraud probability $p \ge \max(0.70, \text{Threshold})$ are categorized as **High Risk**; transactions with $p \ge \text{Threshold}$ are categorized as **Medium Risk**; otherwise, they are classified as **Low Risk**.

### Q15: How can this system be deployed to a cloud environment?
**Answer:** The repository includes a multi-stage `Dockerfile`, `docker-compose.yml`, and `Procfile`. It can be deployed with a single command to Docker, Render, Railway, AWS ECS, or Google Cloud Run.

### Q16: How do you handle concept drift in a production fraud system?
**Answer:** In production, fraudsters continually adapt tactics. The system should incorporate automated drift detection (e.g., Kolmogorov-Smirnov test on input features), monitor flag rates, and trigger scheduled retraining pipelines on a rolling window of recent transactions.

### Q17: What is the time complexity of scoring a single transaction?
**Answer:** During inference, scoring involves $O(1)$ feature engineering calculations, $O(D)$ robust scaling across $D=32$ features, and traversing $T$ decision trees of maximum depth $d$, yielding $O(T \cdot d)$ total complexity. In practice, this takes less than 5 milliseconds.

### Q18: What is the false alarm rate in this project and why is it important?
**Answer:** On the 56,746-transaction test set, the champion model produced only 4 false alarms out of 56,651 legitimate transactions (a false alarm rate under 0.01%). Keeping false alarms low is vital to maintain customer trust and avoid declining legitimate purchases.

### Q19: What role does HistGradientBoosting play in this comparison?
**Answer:** `HistGradientBoostingClassifier` bins continuous features into integer bins (similar to LightGBM), accelerating tree construction on large datasets and providing a strong boosting alternative to bagging.

### Q20: Could deep learning (e.g., MLP or Autoencoders) be used here?
**Answer:** Yes, semi-supervised autoencoders can be trained on legitimate transactions to flag transactions with high reconstruction error as fraud. However, on tabular data with PCA features, tree-based ensembles (Random Forest, XGBoost) consistently match or exceed neural network performance with lower computational overhead.

### Q21: What is data leakage and how would pre-splitting oversampling cause it?
**Answer:** If techniques like SMOTE or random oversampling are applied to the entire dataset before train/test splitting, synthetic copies of test samples will exist in the training set, causing inflated, invalid evaluation scores.

### Q22: What are the primary dependencies of the project?
**Answer:** `scikit-learn` for machine learning algorithms, `pandas` and `numpy` for data manipulation, `matplotlib` and `seaborn` for visualization, `FastAPI` and `uvicorn` for REST APIs, `Streamlit` for dashboarding, and `pytest` for automated unit testing.

### Q23: Why did you implement both Streamlit and FastAPI?
**Answer:** They serve distinct purposes: Streamlit is an interactive, analytical dashboard for data exploration and retraining by human operators; FastAPI is an enterprise-grade REST service designed for low-latency machine-to-machine integrations with core banking payment gateways.

### Q24: What is the role of `confusion_matrix.png` in evaluation?
**Answer:** It provides an explicit visual count of True Negatives, False Positives (False Alarms), False Negatives (Missed Frauds), and True Positives (Detected Frauds), allowing non-technical banking stakeholders to immediately understand the model's operational impact.

### Q25: How would you scale this to handle 5,000 transactions per second?
**Answer:** We would deploy multiple containerized FastAPI instances behind a load balancer (such as NGINX or AWS ALB), run workers asynchronously with Uvicorn, utilize Redis for caching transaction velocity features, and optionally convert the scikit-learn model to ONNX runtime for sub-millisecond C++ inference.

---

## 8. Conclusion
This project successfully demonstrates an end-to-end, statistically sound, and deployable Machine Learning system for credit card fraud detection. By addressing class imbalance through class weighting and threshold tuning on validation data, the model catches the vast majority of fraudulent transactions while maintaining negligible false alarm rates. The resulting dual deployment via FastAPI and Streamlit provides a complete blueprint for both human analysis and automated banking integration.
