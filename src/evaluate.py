"""
Evaluation metrics, confusion matrix computation, and ablation studies.
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score, recall_score, f1_score, precision_score,
    accuracy_score, confusion_matrix, roc_curve, precision_recall_curve
)


def evaluate_predictions(y_true, probs, threshold=0.5):
    """
    Computes all standard classification metrics given true binary labels and predicted probabilities.
    """
    preds = (probs >= threshold).astype(int)
    
    auc = roc_auc_score(y_true, probs)
    recall = recall_score(y_true, preds, pos_label=1, zero_division=0)
    macro_f1 = f1_score(y_true, preds, average='macro', zero_division=0)
    fraud_f1 = f1_score(y_true, preds, pos_label=1, zero_division=0)
    precision = precision_score(y_true, preds, pos_label=1, zero_division=0)
    acc = accuracy_score(y_true, preds)
    
    cm = confusion_matrix(y_true, preds)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = 0, 0, 0, 0

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        'auc': float(auc),
        'recall': float(recall),
        'macro_f1': float(macro_f1),
        'fraud_f1': float(fraud_f1),
        'precision': float(precision),
        'accuracy': float(acc),
        'specificity': float(specificity),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp),
        'threshold': float(threshold)
    }


def find_optimal_threshold(y_true, probs, metric='macro_f1', thresholds=np.linspace(0.1, 0.9, 81)):
    """
    Sweeps thresholds to find the optimal decision boundary on validation set.
    """
    best_score = -1.0
    best_th = 0.5
    for th in thresholds:
        preds = (probs >= th).astype(int)
        if metric == 'macro_f1':
            score = f1_score(y_true, preds, average='macro', zero_division=0)
        elif metric == 'recall':
            score = recall_score(y_true, preds, pos_label=1, zero_division=0)
        elif metric == 'f1':
            score = f1_score(y_true, preds, pos_label=1, zero_division=0)
        else:
            score = f1_score(y_true, preds, average='macro', zero_division=0)

        if score > best_score:
            best_score = score
            best_th = th
    return float(best_th), float(best_score)


def get_roc_and_pr_curves(y_true, probs):
    """Returns FPR, TPR, Recall, Precision arrays for interactive plotting."""
    fpr, tpr, _ = roc_curve(y_true, probs)
    precision, recall, _ = precision_recall_curve(y_true, probs)
    return {
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
        'recalls': recall.tolist(),
        'precisions': precision.tolist()
    }
