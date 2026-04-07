"""Tests for src/utils/metrics.py"""

import numpy as np
import pytest

from src.utils.metrics import (
    compute_metrics,
    physionet_utility_score,
    expected_calibration_error,
    abstention_curve,
    full_evaluation,
)


@pytest.fixture
def perfect_preds():
    y = np.array([0, 0, 1, 0, 1, 1, 0, 0, 1, 0], dtype=np.float32)
    prob = y.copy()
    return y, prob


@pytest.fixture
def random_preds(rng):
    y = (rng.random(100) < 0.1).astype(np.float32)
    prob = rng.random(100).astype(np.float32)
    return y, prob


# ── compute_metrics ───────────────────────────────────────────────────────────

class TestComputeMetrics:
    def test_perfect_auroc(self, perfect_preds):
        y, prob = perfect_preds
        m = compute_metrics(y, prob)
        assert m["auroc"] == pytest.approx(1.0)

    def test_perfect_f1(self, perfect_preds):
        y, prob = perfect_preds
        m = compute_metrics(y, prob, threshold=0.5)
        assert m["f1"] == pytest.approx(1.0)

    def test_keys_present(self, random_preds):
        y, prob = random_preds
        m = compute_metrics(y, prob)
        assert {"auroc", "auprc", "f1", "brier"} <= set(m.keys())

    def test_auroc_in_range(self, random_preds):
        y, prob = random_preds
        m = compute_metrics(y, prob)
        assert 0.0 <= m["auroc"] <= 1.0

    def test_brier_in_range(self, random_preds):
        y, prob = random_preds
        m = compute_metrics(y, prob)
        assert 0.0 <= m["brier"] <= 1.0

    def test_all_negative_warns(self):
        """sklearn warns (does not raise) when only one class present in y_true."""
        import warnings
        y = np.zeros(50, dtype=np.float32)
        prob = np.random.rand(50).astype(np.float32)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            compute_metrics(y, prob)
        assert any("one class" in str(warning.message).lower() for warning in w)


# ── physionet_utility_score ───────────────────────────────────────────────────

class TestPhysionetUtility:
    def test_perfect_predictions(self):
        y = np.array([0, 0, 1, 1, 0], dtype=np.float32)
        pred = y.astype(int)
        score = physionet_utility_score(y, pred)
        assert score == pytest.approx(1.0)

    def test_all_false_positives_penalised(self):
        y = np.zeros(10, dtype=np.float32)
        pred = np.ones(10, dtype=int)
        score = physionet_utility_score(y, pred)
        assert score == 0.0   # max_score=0 → returns 0.0

    def test_all_false_negatives(self):
        y = np.ones(10, dtype=np.float32)
        pred = np.zeros(10, dtype=int)
        score = physionet_utility_score(y, pred)
        assert score < 0.0

    def test_score_in_reasonable_range(self, rng):
        y = (rng.random(200) < 0.1).astype(np.float32)
        pred = (rng.random(200) < 0.1).astype(int)
        score = physionet_utility_score(y, pred)
        assert -10.0 <= score <= 1.0


# ── expected_calibration_error ────────────────────────────────────────────────

class TestECE:
    def test_perfect_calibration(self):
        """If predicted prob == empirical frequency, ECE ≈ 0."""
        # 100 samples: predict 0.1 → 10 % positive
        y = np.array([1] * 10 + [0] * 90, dtype=np.float32)
        prob = np.full(100, 0.1, dtype=np.float32)
        ece = expected_calibration_error(y, prob, n_bins=10)
        assert ece == pytest.approx(0.0, abs=0.02)

    def test_ece_nonnegative(self, random_preds):
        y, prob = random_preds
        ece = expected_calibration_error(y, prob)
        assert ece >= 0.0

    def test_ece_le_one(self, random_preds):
        y, prob = random_preds
        ece = expected_calibration_error(y, prob)
        assert ece <= 1.0


# ── abstention_curve ──────────────────────────────────────────────────────────

class TestAbstentionCurve:
    def test_output_shapes(self, random_preds, rng):
        y, prob = random_preds
        uncertainty = rng.random(len(y)).astype(np.float32)
        cov, prec = abstention_curve(y, prob, uncertainty, n_points=10)
        assert len(cov) == 10
        assert len(prec) == 10

    def test_coverage_ascending(self, random_preds, rng):
        y, prob = random_preds
        uncertainty = rng.random(len(y)).astype(np.float32)
        cov, _ = abstention_curve(y, prob, uncertainty, n_points=10)
        assert list(cov) == sorted(cov)

    def test_coverage_ends_at_one(self, random_preds, rng):
        y, prob = random_preds
        uncertainty = rng.random(len(y)).astype(np.float32)
        cov, _ = abstention_curve(y, prob, uncertainty)
        assert cov[-1] == pytest.approx(1.0)


# ── full_evaluation ───────────────────────────────────────────────────────────

class TestFullEvaluation:
    def test_all_keys_without_uncertainty(self, random_preds):
        y, prob = random_preds
        m = full_evaluation(y, prob)
        assert {"auroc", "auprc", "f1", "brier", "physionet_utility", "ece"} <= set(m.keys())

    def test_auac_key_with_uncertainty(self, random_preds, rng):
        y, prob = random_preds
        uncertainty = rng.random(len(y)).astype(np.float32)
        m = full_evaluation(y, prob, uncertainty)
        assert "auac" in m
        assert 0.0 <= m["auac"] <= 1.0
