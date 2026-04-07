"""
Comparison runner: trains all baselines + loads our method results,
then prints a formatted comparison table.

Baselines:
  1. XGBoost (labelled only)
  2. BiLSTM (labelled only)
  3. Transformer (no augmentation)
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
    try:
        y_true    = np.load("results/y_true.npy")
        mean_prob = np.load("results/mean_prob.npy")
        with open("results/metrics.json") as f:
            metrics = json.load(f)
        return {"y_true": y_true, "mean_prob": mean_prob, "metrics": metrics}
    except FileNotFoundError:
        return None


def run_comparison(splits: dict, cfg: dict, device: torch.device) -> None:
    rows = []

    # ── Baseline 1: XGBoost ───────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("BASELINE 1: XGBoost (labelled only)")
    xgb = train_xgboost(splits, cfg)
    rows.append({"method": "XGBoost (labelled only)", "type": "Supervised",
                 **_fmt(xgb["test"])})

    # ── Baseline 2: BiLSTM ────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("BASELINE 2: BiLSTM (labelled only)")
    lstm = train_lstm(splits, cfg, device)
    rows.append({"method": "BiLSTM (labelled only)", "type": "Supervised",
                 **_fmt(lstm["test"])})

    # ── Baseline 3: Transformer vanilla ──────────────────────────────────────
    logger.info("=" * 60)
    logger.info("BASELINE 3: Transformer (no augmentation)")
    vanilla = train_vanilla_classifier(splits, cfg, device)
    rows.append({"method": "Transformer (no augment)", "type": "Diffusion (no semi-sup)",
                 **_fmt(vanilla["test"])})

    # ── Baseline 4: TimeGAN semi-supervisé ────────────────────────────────────
    logger.info("=" * 60)
    logger.info("BASELINE 4: TimeGAN semi-supervisé (Yoon et al., NeurIPS 2019)")
    timegan = train_timegan(splits, cfg, device)
    rows.append({"method": "TimeGAN (semi-sup.)", "type": "GAN-based",
                 **_fmt(timegan["test"])})

    # ── Baseline 5: Path Signatures ───────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("BASELINE 5: Path Signatures + XGBoost (Morrill et al., 2020)")
    sig = train_signature(splits, cfg, depth=3)
    rows.append({"method": "Path Signatures + XGBoost", "type": "Challenge winner",
                 **_fmt(sig["test"])})

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
        "utility": float("nan"),
        "ece":     float("nan"),
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
            f"{r['auroc']:.4f}",
            f"{r['auprc']:.4f}",
            f"{r['f1']:.4f}",
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
