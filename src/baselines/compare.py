"""
Script de comparaison : entraîne tous les modèles de base + charge les résultats de notre méthode,
puis affiche un tableau comparatif formaté.

Modèles de base :
  1. XGBoost (uniquement étiquetés)
  2. BiLSTM (uniquement étiquetés)
  3. Transformer (sans augmentation)
  4. TimeGAN semi-supervisé (Yoon et al., NeurIPS 2019)
  5. Path Signatures + XGBoost (Morrill et al., 2020 — gagnant PhysioNet 2019)
  ★ Notre méthode : Diffusion-TS + Aug + MC Dropout
"""

from __future__ import annotations

import json
import logging

import numpy as np
import torch

from .xgboost_baseline import train_xgboost
from .lstm_baseline import train_lstm
from .diffusion_vanilla import train_vanilla_classifier
from .timegan_baseline import train_timegan
from .signature_baseline import train_signature

logger = logging.getLogger(__name__)


def _load_our_results() -> dict | None:
    """Charge les résultats de notre méthode (métriques et probabilités) sauvegardés."""
    try:
        y_true    = np.load("results/y_true.npy")
        mean_prob = np.load("results/mean_prob.npy")
        with open("results/metrics.json") as f:
            metrics = json.load(f)
        return {"y_true": y_true, "mean_prob": mean_prob, "metrics": metrics}
    except FileNotFoundError:
        return None


def run_comparison(splits: dict, cfg: dict, device: torch.device) -> None:
    """Exécute l'ensemble des modèles de base et affiche le tableau comparatif final."""
    rows = []

    # ── Modèle de base 1 : XGBoost ───────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("BASELINE 1: XGBoost (labelled only)")
    xgb = train_xgboost(splits, cfg)
    rows.append({"method": "XGBoost (labelled only)", "type": "Supervised",
                 **_fmt(xgb["test"])})

    # ── Modèle de base 2 : BiLSTM ────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("BASELINE 2: BiLSTM (labelled only)")
    lstm = train_lstm(splits, cfg, device)
    rows.append({"method": "BiLSTM (labelled only)", "type": "Supervised",
                 **_fmt(lstm["test"])})

    # ── Modèle de base 3 : Transformer classique ─────────────────────────────────
    logger.info("=" * 60)
    logger.info("BASELINE 3: Transformer (no augmentation)")
    vanilla = train_vanilla_classifier(splits, cfg, device)
    rows.append({"method": "Transformer (no augment)", "type": "Diffusion (no semi-sup)",
                 **_fmt(vanilla["test"])})

    # ── Modèle de base 4 : TimeGAN semi-supervisé ────────────────────────────────
    logger.info("=" * 60)
    logger.info("BASELINE 4: TimeGAN semi-supervisé (Yoon et al., NeurIPS 2019)")
    timegan = train_timegan(splits, cfg, device)
    rows.append({"method": "TimeGAN (semi-sup.)", "type": "GAN-based",
                 **_fmt(timegan["test"])})

    # ── Modèle de base 5 : Path Signatures ───────────────────────────────────────
    logger.info("=" * 60)
    logger.info("BASELINE 5: Path Signatures + XGBoost (Morrill et al., 2020)")
    try:
        sig = train_signature(splits, cfg, depth=2)
        rows.append({"method": "Path Signatures + XGBoost", "type": "Challenge winner",
                     **_fmt(sig["test"])})
    except ImportError as e:
        logger.warning(f"Path Signatures baseline skipped: {e}")
        rows.append({
            "method":  "Path Signatures + XGBoost",
            "type":    "Challenge winner (skipped)",
            "auroc":   float("nan"),
            "auprc":   float("nan"),
            "f1":      float("nan"),
            "utility": float("nan"),
            "ece":     float("nan"),
        })

    # ── Notre méthode ─────────────────────────────────────────────────────────
    our = _load_our_results()
    if our is not None:
        m = our["metrics"]
        rows.append({
            "method":  "Diffusion-TS + Aug + MC Dropout",
            "type":    "Ours (semi-supervised)",
            "auroc":   m["auroc"],
            "auprc":   m["auprc"],
            "f1":      m["f1"],
            "utility": m["physionet_utility"],
            "ece":     m.get("ece", float("nan")),
        })
    else:
        logger.warning("Our results not found — run --stage evaluate first.")

    _print_table(rows)


def _fmt(metrics: dict) -> dict:
    return {
        "auroc":   metrics["auroc"],
        "auprc":   metrics["auprc"],
        "f1":      metrics["f1"],
        "utility": metrics.get("physionet_utility", float("nan")),
        "ece":     metrics.get("ece", float("nan")),
    }


def _print_table(rows: list[dict]) -> None:
    col_w = [38, 24, 8, 8, 8, 10, 8]
    headers = ["Method", "Type", "AUROC", "AUPRC", "F1", "Util", "ECE"]

    sep = "+" + "+".join("-" * w for w in col_w) + "+"
    print("\n" + sep)
    print("|" + "|".join(h.center(w) for h, w in zip(headers, col_w)) + "|")
    print(sep)

    for r in rows:
        marker = " ★" if "Ours" in r["type"] else "  "
        vals = [
            r["method"] + marker,
            r["type"],
            f"{r['auroc']:.4f}"   if not _isnan(r["auroc"])   else "  —  ",
            f"{r['auprc']:.4f}"   if not _isnan(r["auprc"])   else "  —  ",
            f"{r['f1']:.4f}"      if not _isnan(r["f1"])      else "  —  ",
            f"{r['utility']:.4f}" if not _isnan(r["utility"]) else "  —  ",
            f"{r['ece']:.4f}"     if not _isnan(r["ece"])     else "  —  ",
        ]
        print("|" + "|".join(
            v.ljust(w) if i < 2 else v.center(w)
            for i, (v, w) in enumerate(zip(vals, col_w))
        ) + "|")

    print(sep)
    print("  ★ = notre méthode  |  Util = PhysioNet utility score\n")


def _isnan(v) -> bool:
    try:
        return v != v
    except Exception:
        return False
