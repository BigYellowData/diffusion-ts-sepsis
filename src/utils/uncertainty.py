"""
Uncertainty quantification utilities.

Provides:
  - Full test-set MC Dropout evaluation with uncertainty scores
  - Uncertainty calibration analysis
  - Plotting helpers (saved to disk)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..models.classifier import SepsisClassifier, mc_predict
from .metrics import full_evaluation, optimal_threshold

logger = logging.getLogger(__name__)


def _collect_mc_probs(
    classifier: SepsisClassifier,
    loader: DataLoader,
    device: torch.device,
    n_mc_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_probs, all_uncertainties, all_labels = [], [], []
    for batch in loader:
        x = batch["x"].to(device)
        mask = batch["mask"].to(device)
        y = batch["y"].numpy()
        out = mc_predict(classifier, x, mask, n_samples=n_mc_samples)
        all_probs.append(out["mean_prob"].cpu().numpy())
        all_uncertainties.append(out["uncertainty"].cpu().numpy())
        all_labels.append(y)
    return (
        np.concatenate(all_labels),
        np.concatenate(all_probs),
        np.concatenate(all_uncertainties),
    )


@torch.no_grad()
def evaluate_with_uncertainty(
    classifier: SepsisClassifier,
    loader: DataLoader,
    device: torch.device,
    n_mc_samples: int = 50,
    val_loader: DataLoader | None = None,
) -> dict:
    """
    Run MC Dropout inference on a full DataLoader.

    If val_loader is provided, the decision threshold is calibrated on the
    validation set (Youden's J) before being applied to the test set.

    Returns:
        {
          "y_true"       : (N,)
          "mean_prob"    : (N,)
          "uncertainty"  : (N,)  predictive variance
          "metrics"      : dict from full_evaluation()
        }
    """
    # Calibrate threshold on validation set
    threshold = None
    if val_loader is not None:
        val_y, val_prob, _ = _collect_mc_probs(classifier, val_loader, device, n_mc_samples)
        threshold = optimal_threshold(val_y, val_prob)
        logger.info(f"[MC Dropout] Optimal threshold from val set: {threshold:.4f}")

    y_true, mean_prob, uncertainty = _collect_mc_probs(classifier, loader, device, n_mc_samples)

    metrics = full_evaluation(y_true, mean_prob, uncertainty, threshold=threshold)
    logger.info(
        f"[MC Dropout] AUROC={metrics['auroc']:.4f} | "
        f"AUPRC={metrics['auprc']:.4f} | "
        f"ECE={metrics['ece']:.4f} | "
        f"Utility={metrics['physionet_utility']:.4f}"
    )
    return {
        "y_true": y_true,
        "mean_prob": mean_prob,
        "uncertainty": uncertainty,
        "metrics": metrics,
    }


def uncertainty_correlation(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    uncertainty: np.ndarray,
) -> dict:
    """
    Analyse whether high uncertainty correlates with model errors.
    Returns a dict with:
      - Pearson correlation (uncertainty, error)
      - Mean uncertainty for correct vs. incorrect predictions
    """
    y_pred = (y_prob >= 0.5).astype(int)
    errors = (y_pred != y_true).astype(float)

    corr = float(np.corrcoef(uncertainty, errors)[0, 1])
    unc_correct = float(uncertainty[errors == 0].mean())
    unc_incorrect = float(uncertainty[errors == 1].mean())

    logger.info(
        f"Uncertainty–error correlation: {corr:.3f} | "
        f"Unc(correct)={unc_correct:.4f}, Unc(incorrect)={unc_incorrect:.4f}"
    )
    return {
        "corr_uncertainty_error": corr,
        "uncertainty_correct": unc_correct,
        "uncertainty_incorrect": unc_incorrect,
    }


def save_results(results: dict, out_dir: str = "results") -> None:
    """Save numpy arrays and metric summary to disk."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    np.save(out_path / "y_true.npy", results["y_true"])
    np.save(out_path / "mean_prob.npy", results["mean_prob"])
    np.save(out_path / "uncertainty.npy", results["uncertainty"])

    metrics = {k: v for k, v in results["metrics"].items()
               if not isinstance(v, np.ndarray)}
    import json
    with open(out_path / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Results saved to '{out_dir}/'")
