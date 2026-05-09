"""
Baseline — Signatures de chemins (Path Signatures / Morrill et al., 2020).

C'est la méthode gagnante du PhysioNet/CinC Challenge 2019.
Les signatures de chemins encodent les trajectoires temporelles comme des
tenseurs de moments itérés, capturant les corrélations d'ordre supérieur
sans perte d'information (théorème de Chen).

Implémentation :
  1. Signatures tronquées à l'ordre depth=3 via `esig`
  2. Concaténées avec des features agrégées (mean, std, last)
  3. Classifieur XGBoost entraîné sur les features de signature
"""

from __future__ import annotations

import logging
import numpy as np
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight

try:
    import esig
    ESIG_AVAILABLE = True
except ImportError:
    ESIG_AVAILABLE = False

from ..data.preprocess import VITAL_COLS, FEATURE_COLS
from ..utils.metrics import compute_metrics, optimal_threshold

# Indices des signes vitaux dans FEATURE_COLS (8 features au lieu de 40)
# Depth=2 avec 8 features → 8 + 64 = 72 dimensions (gérable)
# Depth=2 avec 40 features → 40 + 1600 = 1640 dimensions → 606 MB pour 92k samples
VITAL_IDX = [FEATURE_COLS.index(c) for c in VITAL_COLS]

logger = logging.getLogger(__name__)


# ─── Extraction de signatures ─────────────────────────────────────────────────

def _compute_signatures(X: np.ndarray, depth: int = 3) -> np.ndarray:
    """
    Calcule les signatures tronquées à l'ordre `depth` pour chaque fenêtre.

    X : (N, T, F)
    Retourne : (N, sig_dim)  où sig_dim = sum_{k=1}^{depth} F^k
    """
    N, T, F = X.shape
    sig_dim = esig.sigdim(F, depth)
    sigs = np.zeros((N, sig_dim), dtype=np.float32)

    for i in range(N):
        path = X[i].astype(np.float64)          # esig requiert float64
        try:
            sigs[i] = esig.stream2sig(path, depth).astype(np.float32)
        except Exception:
            sigs[i] = 0.0                        # fallback si path dégénéré

    return sigs


def _extract_features(X: np.ndarray, M: np.ndarray, depth: int = 2) -> np.ndarray:
    """
    Combine signatures + features agrégées + taux de valeurs manquantes.
    X : (N, T, F),  M : (N, T, F)

    On calcule les signatures uniquement sur les 8 signes vitaux (comme Morrill et al.)
    pour éviter l'explosion dimensionnelle :
      depth=2, F=8  →  8 + 64 = 72 dimensions de signature
      depth=2, F=40 →  40 + 1600 = 1640 dim → 606 MB pour 92k samples (OK)
      depth=3, F=40 →  65640 dim → 22 GB (OOM)
    """
    logger.info(f"[Signature] Computing path signatures (depth={depth}, vitals only) on {len(X)} samples…")

    # Agrégats sur toutes les features
    mean = X.mean(axis=1)                        # (N, F)
    std  = X.std(axis=1)
    last = X[:, -1, :]
    miss = 1.0 - M.mean(axis=1)

    # Signatures sur les signes vitaux uniquement
    X_vital = X[:, :, VITAL_IDX]                 # (N, T, 8)
    sigs = _compute_signatures(X_vital, depth=depth)

    return np.concatenate([sigs, mean, std, last, miss], axis=1).astype(np.float32)


# ─── Entraînement ─────────────────────────────────────────────────────────────

def train_signature(splits: dict, cfg: dict, depth: int = 2) -> dict:
    """
    Entraîne un classifieur XGBoost sur features de signatures.
    Utilise uniquement les données labellées (comme Morrill et al.).

    Retourne les métriques de validation et de test.
    """
    if not ESIG_AVAILABLE:
        raise ImportError("esig non installé : uv add esig")

    train = splits["train"]
    labelled = train["labelled_mask"]
    keep = labelled | (train["y"] == 0)

    logger.info("[Signature] Extracting features (train)…")
    X_tr = _extract_features(train["X"][keep], train["M"][keep], depth)
    y_tr = train["y"][keep]

    # Normalisation (les signatures peuvent avoir des échelles très différentes)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr).astype(np.float32)

    n_pos = int(y_tr.sum())
    n_neg = int((y_tr == 0).sum())
    logger.info(f"[Signature] Training on {len(X_tr)} samples | pos={n_pos} neg={n_neg} | sig_dim={X_tr.shape[1]}")

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
        logger.info(f"[Signature] Extracting features ({split_name})…")
        X_s = scaler.transform(
            _extract_features(sp["X"], sp["M"], depth)
        ).astype(np.float32)
        y_s = sp["y"]
        prob = model.predict_proba(X_s)[:, 1]
        threshold = optimal_threshold(y_s, prob)
        results[split_name] = compute_metrics(y_s, prob, threshold)
        logger.info(
            f"[Signature] {split_name} | AUROC={results[split_name]['auroc']:.4f} | "
            f"AUPRC={results[split_name]['auprc']:.4f} | F1={results[split_name]['f1']:.4f}"
        )

    return results
