from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score,
    f1_score, log_loss, brier_score_loss
)
from scipy import stats

def brier_multiclass(y_true: np.ndarray, proba: np.ndarray, classes: np.ndarray) -> float:
    # sum_k (p_k - y_k)^2 / N
    K = len(classes)
    Y = np.zeros((len(y_true), K), dtype=float)
    for i, c in enumerate(classes):
        Y[:, i] = (y_true == c).astype(float)
    return float(np.mean(np.sum((proba - Y)**2, axis=1)))

def expected_calibration_error(y_true: np.ndarray, proba: np.ndarray, n_bins:int = 15) -> float:
    # binary ECE (use class 1) or max class for multiclass approximation
    p = proba[:, -1] if proba.shape[1] == 2 else proba.max(axis=1)
    bins = np.linspace(0.0, 1.0, n_bins+1)
    idx = np.digitize(p, bins) - 1
    ece = 0.0
    for b in range(n_bins):
        mask = (idx == b)
        if mask.sum() == 0: continue
        conf = p[mask].mean()
        acc = (y_true[mask] == (proba[mask].argmax(axis=1))).mean()
        ece += (mask.mean()) * abs(acc - conf)
    return float(ece)

def compute_metrics(y_true: np.ndarray,
                    proba: np.ndarray,
                    pred: np.ndarray,
                    task: str,
                    classes: np.ndarray) -> Dict[str, float]:
    metrics = {}
    if task == "binary":
        if proba.shape[1] == 2:
            metrics["roc_auc"] = roc_auc_score(y_true, proba[:,1])
            metrics["pr_auc"] = average_precision_score(y_true, proba[:,1])
            metrics["brier"] = brier_score_loss(y_true, proba[:,1])
        else:
            raise ValueError("Binary task expects 2-class probabilities.")
        metrics["accuracy"] = accuracy_score(y_true, pred)
        metrics["precision"] = precision_score(y_true, pred, zero_division=0)
        metrics["recall"] = recall_score(y_true, pred, zero_division=0)
        metrics["f1"] = f1_score(y_true, pred, zero_division=0)
        metrics["logloss"] = log_loss(y_true, proba)
        metrics["ece"] = expected_calibration_error(y_true, proba, n_bins=15)
    else:
        # multiclass
        metrics["roc_auc_macro"] = roc_auc_score(y_true, proba, multi_class="ovr", average="macro")
        metrics["roc_auc_weighted"] = roc_auc_score(y_true, proba, multi_class="ovr", average="weighted")
        metrics["pr_auc_macro"] = average_precision_score(
            y_true, proba, average="macro"
        )
        metrics["accuracy"] = accuracy_score(y_true, pred)
        metrics["precision_macro"] = precision_score(y_true, pred, average="macro", zero_division=0)
        metrics["recall_macro"] = recall_score(y_true, pred, average="macro", zero_division=0)
        metrics["f1_macro"] = f1_score(y_true, pred, average="macro", zero_division=0)
        metrics["logloss"] = log_loss(y_true, proba, labels=classes)
        metrics["brier_macro"] = brier_multiclass(y_true, proba, classes)
        metrics["ece"] = expected_calibration_error(y_true, proba, n_bins=15)
    return metrics

def mean_std_ci(xs: List[float], alpha: float = 0.05) -> Tuple[float,float,Tuple[float,float]]:
    arr = np.asarray(xs, dtype=float)
    m = arr.mean()
    s = arr.std(ddof=1) if len(arr) > 1 else 0.0
    if len(arr) > 1:
        t = stats.t.ppf(1 - alpha/2, df=len(arr)-1)
        margin = t * s / np.sqrt(len(arr))
        return float(m), float(s), (float(m - margin), float(m + margin))
    else:
        return float(m), float(s), (float(m), float(m))

def paired_t_test(xs: List[float], ys: List[float]) -> Tuple[float, float]:
    # returns t-statistic and p-value for paired t-test
    t, p = stats.ttest_rel(xs, ys)
    return float(t), float(p)
