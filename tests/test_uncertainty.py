"""Tests for src/utils/uncertainty.py"""

import json
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from src.models.classifier import SepsisClassifier
from src.utils.uncertainty import (
    evaluate_with_uncertainty,
    uncertainty_correlation,
    save_results,
)


@pytest.fixture
def small_classifier(cfg):
    ccfg = cfg["classifier"]
    dcfg = cfg["diffusion"]
    return SepsisClassifier(
        n_features=dcfg["n_features"],
        seq_len=dcfg["seq_len"],
        d_model=ccfg["d_model"],
        n_heads=ccfg["n_heads"],
        n_layers=ccfg["n_layers"],
        d_ff=ccfg["d_ff"],
        dropout=ccfg["dropout"],
    )


@pytest.fixture
def small_loader(synthetic_splits, cfg):
    from src.data.dataset import SepsisDataset
    split = synthetic_splits["test"]
    ds = SepsisDataset(split["X"], split["M"], split["y"])
    return DataLoader(ds, batch_size=8)


class TestEvaluateWithUncertainty:
    def test_output_keys(self, small_classifier, small_loader, device):
        result = evaluate_with_uncertainty(small_classifier, small_loader, device, n_mc_samples=3)
        assert {"y_true", "mean_prob", "uncertainty", "metrics"} <= set(result.keys())

    def test_array_lengths_match(self, small_classifier, small_loader, device, synthetic_splits):
        result = evaluate_with_uncertainty(small_classifier, small_loader, device, n_mc_samples=3)
        N = len(synthetic_splits["test"]["y"])
        assert len(result["y_true"]) == N
        assert len(result["mean_prob"]) == N
        assert len(result["uncertainty"]) == N

    def test_prob_in_01(self, small_classifier, small_loader, device):
        result = evaluate_with_uncertainty(small_classifier, small_loader, device, n_mc_samples=3)
        assert (result["mean_prob"] >= 0).all()
        assert (result["mean_prob"] <= 1).all()

    def test_uncertainty_nonneg(self, small_classifier, small_loader, device):
        result = evaluate_with_uncertainty(small_classifier, small_loader, device, n_mc_samples=3)
        assert (result["uncertainty"] >= 0).all()


class TestUncertaintyCorrelation:
    def test_keys_present(self, rng):
        y = (rng.random(100) < 0.2).astype(np.float32)
        prob = rng.random(100).astype(np.float32)
        unc = rng.random(100).astype(np.float32)
        stats = uncertainty_correlation(y, prob, unc)
        assert "corr_uncertainty_error" in stats
        assert "uncertainty_correct" in stats
        assert "uncertainty_incorrect" in stats

    def test_high_uncertainty_on_errors(self, rng):
        """Samples with errors should have higher uncertainty than correct ones."""
        rng2 = np.random.default_rng(0)
        y    = (rng2.random(200) < 0.3).astype(np.float32)
        # Deliberately wrong on ~30 % of samples
        prob = y.copy()
        flip = rng2.random(200) < 0.3
        prob[flip] = 1.0 - prob[flip]
        # Assign high uncertainty to flipped (wrong) predictions
        unc = np.where(flip, 0.9, 0.05).astype(np.float32)
        stats = uncertainty_correlation(y, prob, unc)
        assert stats["uncertainty_incorrect"] > stats["uncertainty_correct"]


class TestSaveResults:
    def test_files_created(self, tmp_path, rng):
        results = {
            "y_true":      rng.random(50).astype(np.float32),
            "mean_prob":   rng.random(50).astype(np.float32),
            "uncertainty": rng.random(50).astype(np.float32),
            "metrics":     {"auroc": 0.75, "auprc": 0.4, "ece": 0.1},
        }
        save_results(results, out_dir=str(tmp_path))
        assert (tmp_path / "y_true.npy").exists()
        assert (tmp_path / "mean_prob.npy").exists()
        assert (tmp_path / "uncertainty.npy").exists()
        assert (tmp_path / "metrics.json").exists()

    def test_metrics_json_readable(self, tmp_path, rng):
        results = {
            "y_true":      rng.random(10).astype(np.float32),
            "mean_prob":   rng.random(10).astype(np.float32),
            "uncertainty": rng.random(10).astype(np.float32),
            "metrics":     {"auroc": 0.8},
        }
        save_results(results, out_dir=str(tmp_path))
        with open(tmp_path / "metrics.json") as f:
            data = json.load(f)
        assert data["auroc"] == pytest.approx(0.8)
