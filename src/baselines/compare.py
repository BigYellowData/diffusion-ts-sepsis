"""
Comparison runner: trains all baselines + loads our method results,
then prints a formatted comparison table.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch

from .xgboost_baseline import train_xgboost
from .lstm_baseline import train_lstm
from .diffusion_vanilla import train_vanilla_classifier
from ..utils.metrics import physionet_utility_score, optimal_threshold

logger = logging.getLogger(__name__)


def _load_our_results() -> dict | None:
    """Load precomputed results from the evaluate stage."""
    try:
        y_true    = np.load("results/y_true.npy")
        mean_prob = np.load("results/mean_prob.npy")
        with open("results/metrics.json") as f:
            metrics = json.load(f)
        return {"y_true": y_true, "mean_prob": mean_prob, "metrics": metrics}
    except FileNotFoundError:
        return None


def run_comparison(splits: dict, cfg: dict, device: torch.device) -> None:
    """
    Train all baselines, load our method results, print comparison table.
    """
    rows = []

    # ── Baseline 1: XGBoost ───────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("BASELINE 1: XGBoost (labelled only)")
    xgb = train_xgboost(splits, cfg)
    rows.append({
        "method":  "XGBoost (labelled only)",
        "type":    "Supervised",
        **_prefix(xgb["test"]),
    })

    # ── Baseline 2: LSTM ──────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("BASELINE 2: BiLSTM (labelled only)")
    lstm = train_lstm(splits, cfg, device)
    rows.append({
        "method":  "BiLSTM (labelled only)",
        "type":    "Supervised",
        **_prefix(lstm["test"]),
    })

    # ── Baseline 3: Diffusion-TS vanilla (no augmentation) ───────────────────
    logger.info("=" * 60)
    logger.info("BASELINE 3: Diffusion-TS Transformer (no augmentation, no MC Dropout)")
    vanilla = train_vanilla_classifier(splits, cfg, device)
    rows.append({
        "method":  "Transformer (no augment)",
        "type":    "Diffusion (no semi-sup)",
        **_prefix(vanilla["test"]),
    })

    # ── Our method ────────────────────────────────────────────────────────────
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
        logger.warning("Our method results not found — run --stage evaluate first.")

    # ── Print table ───────────────────────────────────────────────────────────
    _print_table(rows)


def _prefix(metrics: dict) -> dict:
    """Extract the relevant subset of metrics and add physionet utility."""
    from ..utils.metrics import physionet_utility_score
    import numpy as np

    y_pred = None
    return {
        "auroc":   metrics["auroc"],
        "auprc":   metrics["auprc"],
        "f1":      metrics["f1"],
        "utility": float("nan"),   # not recomputed here (no raw preds)
        "ece":     float("nan"),
    }


def _print_table(rows: list[dict]) -> None:
    col_w = [38, 22, 8, 8, 8, 10, 8]
    headers = ["Method", "Type", "AUROC", "AUPRC", "F1", "Util", "ECE"]

    sep = "+" + "+".join("-" * w for w in col_w) + "+"
    header_row = "|" + "|".join(
        h.center(w) for h, w in zip(headers, col_w)
    ) + "|"

    print("\n" + sep)
    print(header_row)
    print(sep)

    for r in rows:
        vals = [
            r["method"],
            r["type"],
            f"{r['auroc']:.4f}",
            f"{r['auprc']:.4f}",
            f"{r['f1']:.4f}",
            f"{r['utility']:.4f}" if not _isnan(r["utility"]) else "  —  ",
            f"{r['ece']:.4f}"     if not _isnan(r["ece"])     else "  —  ",
        ]
        # Highlight our method
        marker = " ★" if "Ours" in r["type"] else "  "
        vals[0] = vals[0] + marker
        print("|" + "|".join(v.ljust(w) if i < 2 else v.center(w)
                             for i, (v, w) in enumerate(zip(vals, col_w))) + "|")

    print(sep)
    print("  ★ = our method   |   Util = PhysioNet utility score\n")


def _isnan(v) -> bool:
    try:
        return v != v  # nan != nan
    except Exception:
        return False
