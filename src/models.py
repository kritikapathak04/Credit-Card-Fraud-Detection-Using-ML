"""Model definitions for Credit Card Fraud Detection.

Compares:
1. Logistic Regression (linear baseline)
2. Random Forest (bagging ensemble)
3. HistGradientBoosting (boosting ensemble)
4. Isolation Forest (unsupervised anomaly detection baseline)
"""
from typing import Dict

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression


class IsolationForestClassifier(BaseEstimator, ClassifierMixin):
    """Scikit-learn compatible classifier wrapper around IsolationForest

    to allow standard scoring, thresholding, and pipeline evaluation.
    """

    def __init__(self, contamination: float = 0.002, random_state: int = 42):
        self.contamination = contamination
        self.random_state = random_state
        self.model_ = None
        self.classes_ = np.array([0, 1])

    def fit(self, X, y=None):
        self.model_ = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1,
        )
        # Isolation Forest is unsupervised; fits on features only
        self.model_.fit(X)
        return self

    def decision_function(self, X):
        # In scikit-learn, lower decision_function values = anomalies.
        # We invert so higher values = higher likelihood of fraud.
        raw_scores = self.model_.decision_function(X)
        return -raw_scores

    def predict_proba(self, X):
        # Convert inverted anomaly scores to pseudo-probabilities via sigmoid
        scores = self.decision_function(X)
        probs_fraud = 1.0 / (1.0 + np.exp(-3.0 * scores))
        probs_fraud = np.clip(probs_fraud, 0.0, 1.0)
        return np.column_stack([1.0 - probs_fraud, probs_fraud])

    def predict(self, X):
        probs = self.predict_proba(X)[:, 1]
        return (probs >= 0.5).astype(int)


def get_candidate_models(
    fast: bool = False, random_state: int = 42
) -> Dict[str, BaseEstimator]:
    """Return dictionary of the four candidate models configured for imbalanced learning."""
    rf_trees = 40 if fast else 100
    gb_iters = 40 if fast else 100

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=rf_trees,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
        "Gradient Boosting": HistGradientBoostingClassifier(
            class_weight="balanced",
            max_iter=gb_iters,
            random_state=random_state,
        ),
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=random_state,
        ),
        "Isolation Forest": IsolationForestClassifier(
            contamination=0.002,
            random_state=random_state,
        ),
    }
    return models
