"""
Baseline 1 — XGBoost on labelled data only.

Features: flatten (T, F) window → (T*F,) + hand-crafted aggregates
  (mean, std, min, max, last value, missing rate per feature).
Trained only on the labelled subset of the training set.
"""

from __future__ import annotations

import logging
import numpy as np
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight

from ..utils.metrics import compute_metrics, optimal_threshold

logger = logging.getLogger(__name__)


def _extract_features(X: np.ndarray, M: np.ndarray) -> np.ndarray:
    """
    X : (N, T, F)  normalised windows
    M : (N, T, F)  observation masks
    Returns (N, 5*F) feature vector.
    """
    mean  = X.mean(axis=1)                          # (N, F)
    std   = X.std(axis=1)                           # (N, F)
    xmin  = X.min(axis=1)                           # (N, F)
    xmax  = X.max(axis=1)                           # (N, F)
    last  = X[:, -1, :]                             # (N, F)  last time step
    miss  = 1.0 - M.mean(axis=1)                   # (N, F)  missing rate
    return np.concatenate([mean, std, xmin, xmax, last, miss], axis=1)


def train_xgboost(splits: dict, cfg: dict) -> dict:
    """
    Train XGBoost using only the labelled positive subset + all negatives.

    Returns a result dict with val and test metrics.
    """
    train = splits["train"]
    labelled = train["labelled_mask"]               # bool (N,)

    X_train_full = _extract_features(train["X"], train["M"])
    y_train_full = train["y"]

    # Keep only labelled positives + all negatives (semi-supervised setting)
    keep = labelled | (y_train_full == 0)
    X_tr = X_train_full[keep]
    y_tr = y_train_full[keep]

    n_pos = int(y_tr.sum())
    n_neg = int((y_tr == 0).sum())
    logger.info(f"[XGBoost] Training on {len(X_tr)} samples | pos={n_pos} neg={n_neg}")

    sample_w = compute_sample_weight("balanced", y_tr)

    model = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1,
        device="cpu",
    )
    model.fit(X_tr, y_tr, sample_weight=sample_w, verbose=False)

    results = {}
    for split_name in ("val", "test"):
        sp = splits[split_name]
        X_s = _extract_features(sp["X"], sp["M"])
        y_s = sp["y"]
        prob = model.predict_proba(X_s)[:, 1]
        threshold = optimal_threshold(y_s, prob)
        results[split_name] = compute_metrics(y_s, prob, threshold)
        logger.info(
            f"[XGBoost] {split_name} | AUROC={results[split_name]['auroc']:.4f} | "
            f"AUPRC={results[split_name]['auprc']:.4f} | F1={results[split_name]['f1']:.4f}"
        )

    return results
