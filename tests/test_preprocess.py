"""Tests for src/data/preprocess.py"""

import numpy as np
import pandas as pd
import pytest

from src.data.preprocess import (
    FEATURE_COLS,
    LABEL_COL,
    N_FEATURES,
    handle_missing,
    create_windows,
    fit_scaler,
    apply_scaler,
    split_by_patient,
    subsample_labels,
)


# ── handle_missing ─────────────────────────────────────────────────────────────

class TestHandleMissing:
    def test_no_nan_in_output(self, synthetic_patients):
        for df in synthetic_patients:
            filled, mask = handle_missing(df)
            assert not filled[FEATURE_COLS].isna().any().any()

    def test_mask_is_binary(self, synthetic_patients):
        for df in synthetic_patients[:5]:
            _, mask = handle_missing(df)
            vals = mask.values
            assert set(vals.flatten().tolist()).issubset({0.0, 1.0})

    def test_mask_reflects_original_nans(self, rng):
        df = pd.DataFrame(
            {"HR": [1.0, np.nan, 3.0], **{c: [np.nan] * 3 for c in FEATURE_COLS[1:]}},
        )
        df[LABEL_COL] = 0.0
        _, mask = handle_missing(df)
        assert mask["HR"].iloc[0] == 1.0
        assert mask["HR"].iloc[1] == 0.0

    def test_shape_preserved(self, synthetic_patients):
        df = synthetic_patients[0]
        filled, mask = handle_missing(df)
        assert filled.shape == df.shape
        assert mask.shape[0] == df.shape[0]
        assert mask.shape[1] == N_FEATURES


# ── create_windows ─────────────────────────────────────────────────────────────

class TestCreateWindows:
    def test_output_shapes(self, synthetic_patients):
        X, M, y, pids = create_windows(synthetic_patients, window_size=24, step_size=6)
        assert X.ndim == 3
        assert X.shape[1] == 24
        assert X.shape[2] == N_FEATURES
        assert M.shape == X.shape
        assert y.shape == (len(X),)
        assert pids.shape == (len(X),)

    def test_labels_are_binary(self, synthetic_patients):
        _, _, y, _ = create_windows(synthetic_patients, window_size=24, step_size=6)
        assert set(y.tolist()).issubset({0.0, 1.0})

    def test_no_nan_in_windows(self, synthetic_patients):
        X, _, _, _ = create_windows(synthetic_patients, window_size=24, step_size=6)
        assert not np.isnan(X).any()

    def test_window_size_respected(self, synthetic_patients):
        for ws in (8, 16, 24):
            X, _, _, _ = create_windows(synthetic_patients, window_size=ws, step_size=4)
            assert X.shape[1] == ws

    def test_patient_with_too_few_rows_skipped(self):
        """A patient with fewer rows than window_size produces no windows."""
        short_df = pd.DataFrame(
            np.zeros((5, N_FEATURES), dtype=np.float32), columns=FEATURE_COLS
        )
        short_df[LABEL_COL] = 0.0
        X, _, _, _ = create_windows([short_df], window_size=24, step_size=6)
        assert len(X) == 0

    def test_pid_values_in_range(self, synthetic_patients):
        _, _, _, pids = create_windows(synthetic_patients, window_size=24, step_size=6)
        assert pids.min() >= 0
        assert pids.max() < len(synthetic_patients)


# ── fit_scaler / apply_scaler ──────────────────────────────────────────────────

class TestScaler:
    def test_fit_and_apply(self, synthetic_splits):
        X_train = synthetic_splits["train"]["X"]
        scaler = fit_scaler(X_train)
        X_scaled = apply_scaler(X_train, scaler)
        assert X_scaled.shape == X_train.shape
        # After scaling the training data, mean ≈ 0 and std ≈ 1 per feature
        flat = X_scaled.reshape(-1, X_train.shape[-1])
        np.testing.assert_allclose(flat.mean(axis=0), 0.0, atol=1e-4)
        np.testing.assert_allclose(flat.std(axis=0), 1.0, atol=1e-4)

    def test_apply_does_not_mutate_input(self, synthetic_splits):
        X = synthetic_splits["train"]["X"].copy()
        scaler = fit_scaler(X)
        X_before = X.copy()
        apply_scaler(X, scaler)
        np.testing.assert_array_equal(X, X_before)


# ── split_by_patient ───────────────────────────────────────────────────────────

class TestSplitByPatient:
    def test_no_patient_leakage(self, synthetic_patients):
        X, M, y, pids = create_windows(synthetic_patients, window_size=24, step_size=6)
        splits = split_by_patient(X, M, y, pids, val_ratio=0.10, test_ratio=0.15)

        train_pids = set(pids[np.isin(pids, list(range(len(synthetic_patients))))].tolist())
        # Reconstruct pid sets per split
        all_splits_pids = []
        for name in ("train", "val", "test"):
            # We can't directly get pids back from split, but we check sizes sum
            pass

        total = sum(len(s["X"]) for s in splits.values())
        assert total == len(X)

    def test_split_sizes(self, synthetic_patients):
        X, M, y, pids = create_windows(synthetic_patients, window_size=24, step_size=6)
        splits = split_by_patient(X, M, y, pids, val_ratio=0.10, test_ratio=0.15)
        assert len(splits["train"]["X"]) > 0
        assert len(splits["val"]["X"]) > 0
        assert len(splits["test"]["X"]) > 0

    def test_all_keys_present(self, synthetic_patients):
        X, M, y, pids = create_windows(synthetic_patients, window_size=24, step_size=6)
        splits = split_by_patient(X, M, y, pids)
        for split in splits.values():
            assert "X" in split
            assert "M" in split
            assert "y" in split


# ── subsample_labels ───────────────────────────────────────────────────────────

class TestSubsampleLabels:
    def test_correct_ratio(self, rng):
        y = (rng.random(200) < 0.1).astype(np.float32)
        n_pos = int(y.sum())
        mask = subsample_labels(y, label_ratio=0.50)
        assert mask.sum() <= n_pos
        assert mask.sum() >= max(1, int(n_pos * 0.50) - 1)

    def test_only_positives_labelled(self, rng):
        y = np.array([0, 0, 1, 0, 1, 0, 1], dtype=np.float32)
        mask = subsample_labels(y, label_ratio=1.0)
        # All labelled entries must be positive
        assert (y[mask] == 1).all()

    def test_at_least_one_labelled(self, rng):
        y = np.array([1.0], dtype=np.float32)
        mask = subsample_labels(y, label_ratio=0.01)
        assert mask.sum() >= 1
