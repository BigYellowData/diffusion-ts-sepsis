"""
Métriques d'évaluation pour la prédiction du sepsis.

Implémente :
  - AUROC / AUPRC / F1
  - Score d'utilité PhysioNet 2019
  - ECE (Erreur de calibration attendue / Expected Calibration Error)
  - Courbe d'abstention (précision vs couverture)
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


# ─── Métriques de classification standard ─────────────────────────────────────────

def optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Trouve le seuil qui maximise l'indice J de Youden (sensibilité + spécificité - 1)."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j_scores = tpr - fpr
    return float(thresholds[np.argmax(j_scores)])


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float | None = None,
) -> dict:
    """
    Calcule l'AUROC, l'AUPRC, le F1, le score de Brier, l'ECE, et l'utilité PhysioNet.
    Si le seuil est None, le seuil optimal de Youden est utilisé automatiquement.
    """
    if threshold is None:
        threshold = optimal_threshold(y_true, y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "auroc": roc_auc_score(y_true, y_prob),
        "auprc": average_precision_score(y_true, y_prob),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "brier": brier_score_loss(y_true, y_prob),
        "ece": expected_calibration_error(y_true, y_prob),
        "physionet_utility": physionet_utility_score(y_true, y_pred),
        "threshold": threshold,
    }


# ─── Score d'utilité PhysioNet 2019 ─────────────────────────────────────────────

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
    Score d'utilité simplifié de PhysioNet 2019.

    Le score réel est calculé par patient sur l'ensemble de la série temporelle.
    Cette version fonctionne sur les prédictions fenêtrées (approximation).

    Récompenses :
      TP prédit tôt       : +1.0 (jusqu'à 12h avant le déclenchement)
      TP prédit à l'heure : +1.0 (6h avant → déclenchement)
      TP prédit en retard : décroissance linéaire vers -2 (FN)
      FP                  : -0.05 par fausse alarme
      FN                  : -2.0
      TN                  : 0.0
    """
    positives = y_true == 1
    negatives = y_true == 0

    # Approximation : traite tôt/optimal/en retard comme un seul groupe TP
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
    # Normalise par le score maximum possible (tous les TP)
    max_score = positives.sum() * max_u_tp
    if max_score == 0:
        return 0.0
    return float(score / max_score)


# ─── Erreur de calibration attendue (ECE) ───────────────────────────────────────────────

def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    ECE : moyenne pondérée de |précision - confiance| par intervalle de confiance.
    Plus la valeur est faible, mieux c'est.
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


# ─── Courbe d'abstention ─────────────────────────────────────────────────────────

def abstention_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    uncertainty: np.ndarray,
    n_points: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calcule la précision à différents niveaux de couverture lorsque le modèle s'abstient
    sur ses prédictions les plus incertaines.

    Retourne :
        coverages   – fraction d'échantillons conservés (incertitude décroissante supprimée)
        precisions  – précision sur les échantillons conservés
    """
    order = np.argsort(uncertainty)          # les plus certains d'abord
    coverages, precisions = [], []

    for k in np.linspace(0.1, 1.0, n_points):
        n_keep = max(1, int(k * len(y_true)))
        idx = order[:n_keep]
        y_pred_k = (y_prob[idx] >= 0.5).astype(int)
        # Précision = TP / (TP + FP)
        tp = ((y_pred_k == 1) & (y_true[idx] == 1)).sum()
        fp = ((y_pred_k == 1) & (y_true[idx] == 0)).sum()
        prec = tp / (tp + fp + 1e-9)
        coverages.append(k)
        precisions.append(prec)

    return np.array(coverages), np.array(precisions)


# ─── Suite d'évaluation complète ────────────────────────────────────────────────────

def full_evaluation(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    uncertainty: np.ndarray | None = None,
    threshold: float | None = None,
) -> dict:
    """Exécute toutes les métriques et retourne un dictionnaire récapitulatif."""
    results = compute_metrics(y_true, y_prob, threshold)

    if uncertainty is not None:
        cov, prec = abstention_curve(y_true, y_prob, uncertainty)
        results["abstention_coverage"] = cov
        results["abstention_precision"] = prec
        # Aire sous la courbe d'abstention
        results["auac"] = float(np.trapezoid(prec, cov))

    return results
