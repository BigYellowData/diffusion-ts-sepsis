"""
Shared fixtures used across all test modules.
All fixtures use synthetic in-memory data — no disk access required.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.preprocess import FEATURE_COLS, LABEL_COL


# ── Sizes used throughout ──────────────────────────────────────────────────────
N_FEATURES = len(FEATURE_COLS)   # 40
WINDOW_SIZE = 24
N_PATIENTS  = 20
N_WINDOWS   = 60   # total synthetic windows


# ── Random-state fixture ───────────────────────────────────────────────────────

@pytest.fixture
def rng():
    return np.random.default_rng(42)


# ── Synthetic patient DataFrames ───────────────────────────────────────────────

@pytest.fixture
def synthetic_patients(rng):
    """
    List of N_PATIENTS small DataFrames mimicking PhysioNet .psv files.
    Each patient has between 30 and 50 hourly rows.
    ~10 % of patients are sepsis-positive (SepsisLabel flips to 1 mid-stay).
    ~30 % of feature values are NaN to simulate missing data.
    """
    patients = []
    for i in range(N_PATIENTS):
        T = rng.integers(30, 51)
        data = rng.standard_normal((T, N_FEATURES)).astype(np.float32)

        # Sprinkle NaNs
        nan_mask = rng.random((T, N_FEATURES)) < 0.30
        data[nan_mask] = np.nan

        df = pd.DataFrame(data, columns=FEATURE_COLS)

        # ~10 % patients develop sepsis
        labels = np.zeros(T, dtype=np.float32)
        if rng.random() < 0.10:
            onset = rng.integers(T // 2, T)
            labels[onset:] = 1.0
        df[LABEL_COL] = labels
        patients.append(df)
    return patients


# ── Synthetic numpy splits ─────────────────────────────────────────────────────

@pytest.fixture
def synthetic_splits(rng):
    """
    Pre-built train/val/test splits as numpy arrays,
    bypassing the full preprocessing pipeline.
    """
    def _make(n):
        X = rng.standard_normal((n, WINDOW_SIZE, N_FEATURES)).astype(np.float32)
        M = (rng.random((n, WINDOW_SIZE, N_FEATURES)) > 0.3).astype(np.float32)
        y = (rng.random(n) < 0.08).astype(np.float32)   # ~8 % positive
        return {"X": X, "M": M, "y": y}

    train = _make(N_WINDOWS)
    train["labelled_mask"] = (rng.random(N_WINDOWS) < 0.5).astype(bool)
    return {
        "train": train,
        "val":   _make(16),
        "test":  _make(16),
    }


# ── Minimal config ─────────────────────────────────────────────────────────────

@pytest.fixture
def cfg():
    return {
        "data": {
            "raw_dir":       "data/raw/sepsis",
            "processed_dir": "data/processed",
            "window_size":   WINDOW_SIZE,
            "step_size":     6,
            "label_ratio":   0.10,
            "val_ratio":     0.10,
            "test_ratio":    0.15,
            "seed":          42,
        },
        "diffusion": {
            "n_features":        N_FEATURES,
            "seq_len":           WINDOW_SIZE,
            "d_model":           32,   # tiny for tests
            "n_heads":           4,
            "n_layers":          2,
            "d_ff":              64,
            "dropout":           0.1,
            "n_diffusion_steps": 10,   # very short schedule
            "beta_start":        1e-4,
            "beta_end":          0.02,
            "scheduler":         "cosine",
        },
        "classifier": {
            "d_model":    32,
            "n_heads":    4,
            "n_layers":   2,
            "d_ff":       64,
            "dropout":    0.1,
            "mc_samples": 5,
        },
        "generation": {
            "n_synthetic_per_real": 1,
            "guidance_scale":       1.0,
        },
        "training": {
            "diffusion_epochs":      1,
            "diffusion_lr":          1e-3,
            "diffusion_batch_size":  8,
            "diffusion_warmup_steps": 0,
            "classifier_epochs":     1,
            "classifier_lr":         1e-3,
            "classifier_batch_size": 8,
            "class_weight_beta":     0.9999,
            "checkpoint_dir":        "checkpoints_test",
            "log_dir":               "logs_test",
        },
        "device": "cpu",
    }


# ── Torch device ───────────────────────────────────────────────────────────────

@pytest.fixture
def device():
    return torch.device("cpu")
