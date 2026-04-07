"""
Evaluation metrics for sepsis prediction.

Implements:
  - AUROC / AUPRC / F1
  - PhysioNet 2019 utility score
  - ECE (Expected Calibration Error)
  - Abstention curve (precision vs. coverage)
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_curve,
    brier_score_loss,
)


# ─── Standard classification metrics ─────────────────────────────────────────

def optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Find the threshold that maximises Youden's J (sensitivity + specificity - 1)."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j_scores = tpr - fpr
    return float(thresholds[np.argmax(j_scores)])


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float | None = None,
) -> dict:
    """
    Compute AUROC, AUPRC, F1, Brier score.
    If threshold is None, the optimal Youden threshold is used automatically.
    """
    if threshold is None:
        threshold = optimal_threshold(y_true, y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "auroc": roc_auc_score(y_true, y_prob),
        "auprc": average_precision_score(y_true, y_prob),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "brier": brier_score_loss(y_true, y_prob),
        "threshold": threshold,
    }


# ─── PhysioNet 2019 utility score ─────────────────────────────────────────────

def physionet_utility_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dt_early: int = 12,
    dt_optimal: int = 6,
    dt_late: int = 3,
    max_u_tp: float = 1.0,
    min_u_fp: float = -0.05,
    u_fn: float = -2.0,
    u_tn: float = 0.0,
) -> float:
    """
    Simplified PhysioNet 2019 utility score.

    The true score is computed per-patient over the full time series.
    This version operates on the windowed predictions (approximate).

    Rewards:
      TP predicted early  : +1.0 (up to 12h before onset)
      TP predicted on time: +1.0 (6h before → onset)
      TP predicted late   : linear decay toward −2 (FN)
      FP                  : −0.05 per false alarm
      FN                  : −2.0
      TN                  : 0.0
    """
    positives = y_true == 1
    negatives = y_true == 0

    # Approximate: treat early/optimal/late as a single TP bucket
    tp = (y_pred == 1) & positives
    fp = (y_pred == 1) & negatives
    fn = (y_pred == 0) & positives
    tn = (y_pred == 0) & negatives

    score = (
        tp.sum() * max_u_tp
        + fp.sum() * min_u_fp
        + fn.sum() * u_fn
        + tn.sum() * u_tn
    )
    # Normalise by maximum possible score (all TP)
    max_score = positives.sum() * max_u_tp
    if max_score == 0:
        return 0.0
    return float(score / max_score)


# ─── Expected Calibration Error ───────────────────────────────────────────────

def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    ECE: weighted average of |accuracy - confidence| per confidence bin.
    Lower is better.
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


# ─── Abstention curve ─────────────────────────────────────────────────────────

def abstention_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    uncertainty: np.ndarray,
    n_points: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute precision at different coverage levels when the model abstains
    on its most uncertain predictions.

    Returns:
        coverages   – fraction of samples retained (descending uncertainty removed)
        precisions  – precision on retained samples
    """
    order = np.argsort(uncertainty)          # most certain first
    coverages, precisions = [], []

    for k in np.linspace(0.1, 1.0, n_points):
        n_keep = max(1, int(k * len(y_true)))
        idx = order[:n_keep]
        y_pred_k = (y_prob[idx] >= 0.5).astype(int)
        # Precision = TP / (TP + FP)
        tp = ((y_pred_k == 1) & (y_true[idx] == 1)).sum()
        fp = ((y_pred_k == 1) & (y_true[idx] == 0)).sum()
        prec = tp / (tp + fp + 1e-9)
        coverages.append(k)
        precisions.append(prec)

    return np.array(coverages), np.array(precisions)


# ─── Full evaluation suite ────────────────────────────────────────────────────

def full_evaluation(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    uncertainty: np.ndarray | None = None,
    threshold: float | None = None,
) -> dict:
    """Run all metrics and return a summary dict."""
    if threshold is None:
        threshold = optimal_threshold(y_true, y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    results = compute_metrics(y_true, y_prob, threshold)
    results["physionet_utility"] = physionet_utility_score(y_true, y_pred)
    results["ece"] = expected_calibration_error(y_true, y_prob)

    if uncertainty is not None:
        cov, prec = abstention_curve(y_true, y_prob, uncertainty)
        results["abstention_coverage"] = cov
        results["abstention_precision"] = prec
        # Area under abstention curve
        results["auac"] = float(np.trapezoid(prec, cov))

    return results
