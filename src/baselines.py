"""
Machine Learning Baseline Models for Fraud Detection.
Implements Logistic Regression, Random Forest, and XGBoost with class-weighting and probability calibration.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import roc_auc_score, recall_score, f1_score, precision_score


class BaselineModelWrapper:
    """Standard wrapper for ML baselines with threshold tuning."""
    def __init__(self, name, model, optimal_threshold=0.5):
        self.name = name
        self.model = model
        self.optimal_threshold = optimal_threshold

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        self.model.fit(X_train, y_train)
        if X_val is not None and y_val is not None:
            self.tune_threshold(X_val, y_val)
        return self

    def predict_proba(self, X):
        if hasattr(self.model, 'predict_proba'):
            probs = self.model.predict_proba(X)[:, 1]
        elif hasattr(self.model, 'decision_function'):
            scores = self.model.decision_function(X)
            probs = 1 / (1 + np.exp(-scores))
        else:
            probs = self.model.predict(X).astype(float)
        return probs

    def predict(self, X, threshold=None):
        th = threshold if threshold is not None else self.optimal_threshold
        probs = self.predict_proba(X)
        return (probs >= th).astype(int)

    def tune_threshold(self, X_val, y_val, thresholds=np.linspace(0.1, 0.9, 81)):
        probs = self.predict_proba(X_val)
        best_f1 = -1.0
        best_th = 0.5
        for th in thresholds:
            preds = (probs >= th).astype(int)
            f1 = f1_score(y_val, preds, average='macro', zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_th = th
        self.optimal_threshold = float(best_th)
        return self.optimal_threshold


def build_baselines(pos_weight=10.0, seed=42):
    """
    Constructs the 3 baseline models matching the presentation specification.
    """
    models = {
        'Logistic Regression': BaselineModelWrapper(
            name='Logistic Regression',
            model=LogisticRegression(
                class_weight='balanced',
                max_iter=1000,
                C=1.0,
                solver='lbfgs',
                random_state=seed
            ),
            optimal_threshold=0.45
        ),
        'Random Forest': BaselineModelWrapper(
            name='Random Forest',
            model=RandomForestClassifier(
                n_estimators=150,
                max_depth=12,
                class_weight='balanced_subsample',
                random_state=seed,
                n_jobs=-1
            ),
            optimal_threshold=0.42
        ),
        'XGBoost': BaselineModelWrapper(
            name='XGBoost',
            model=xgb.XGBClassifier(
                n_estimators=180,
                max_depth=6,
                learning_rate=0.08,
                subsample=0.85,
                colsample_bytree=0.85,
                scale_pos_weight=pos_weight,
                eval_metric='logloss',
                random_state=seed,
                n_jobs=-1
            ),
            optimal_threshold=0.40
        )
    }
    return models
