"""
Modèle de base 1 — XGBoost sur les données étiquetées uniquement.

Caractéristiques : aplatissement de la fenêtre (T, F) → (T*F,) + agrégats créés manuellement
  (moyenne, écart-type, min, max, dernière valeur, taux de valeurs manquantes par caractéristique).
Entraîné uniquement sur le sous-ensemble étiqueté de l'ensemble d'entraînement.
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
    X : (N, T, F)  fenêtres normalisées
    M : (N, T, F)  masques d'observation
    Retourne un vecteur de caractéristiques de dimension (N, 6*F).
    """
    mean  = X.mean(axis=1)                          # (N, F)
    std   = X.std(axis=1)                           # (N, F)
    xmin  = X.min(axis=1)                           # (N, F)
    xmax  = X.max(axis=1)                           # (N, F)
    last  = X[:, -1, :]                             # (N, F)  dernière étape temporelle
    miss  = 1.0 - M.mean(axis=1)                   # (N, F)  taux de valeurs manquantes
    return np.concatenate([mean, std, xmin, xmax, last, miss], axis=1)


def train_xgboost(splits: dict, cfg: dict) -> dict:
    """
    Entraîne XGBoost en utilisant uniquement le sous-ensemble positif étiqueté + tous les négatifs.

    Retourne un dictionnaire de résultats contenant les métriques de validation et de test.
    """
    train = splits["train"]
    labelled = train["labelled_mask"]               # bool (N,)

    X_train_full = _extract_features(train["X"], train["M"])
    y_train_full = train["y"]

    # Ne garde que les positifs étiquetés + tous les négatifs (configuration semi-supervisée)
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
