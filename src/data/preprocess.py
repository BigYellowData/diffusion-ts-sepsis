"""
PhysioNet/CinC Challenge 2019 – preprocessing pipeline.

Expected raw data layout:
  data/raw/
    training_setA/   (or training/)
      p000001.psv
      p000002.psv
      ...
    training_setB/   (optional second hospital set)
      ...

Each .psv file has a header row then one row per hour.
Columns: HR, O2Sat, Temp, SBP, MAP, DBP, Resp, EtCO2, ..., SepsisLabel
"""

from __future__ import annotations

import os
import glob
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────

VITAL_COLS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2"]
LAB_COLS = [
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
    "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
    "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium",
    "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC",
    "Fibrinogen", "Platelets",
]
DEMO_COLS = ["Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS"]
FEATURE_COLS = VITAL_COLS + LAB_COLS + DEMO_COLS
LABEL_COL = "SepsisLabel"
N_FEATURES = len(FEATURE_COLS)  # 40


# ─── Loading ──────────────────────────────────────────────────────────────────

def _read_psv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="|", na_values=["NaN"], dtype=np.float32)
    for col in FEATURE_COLS + [LABEL_COL]:
        if col not in df.columns:
            df[col] = np.nan
    return df[FEATURE_COLS + [LABEL_COL]]


def load_raw_patients(raw_dir: str, n_workers: int = 8) -> List[pd.DataFrame]:
    """
    Read all .psv files in parallel and return a list of per-patient DataFrames.

    Handles both sub-structures found in PhysioNet 2019:
      - training_setA/training/*.psv
      - training_setB/training_setB/*.psv
    Missing values are encoded as 'NaN' strings in the raw files.
    """
    pattern = os.path.join(raw_dir, "**", "*.psv")
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        raise FileNotFoundError(
            f"No .psv files found under '{raw_dir}'. "
            "Expected layout: data/raw/sepsis/training_setA/training/*.psv"
        )

    patients = [None] * len(files)
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_read_psv, f): i for i, f in enumerate(files)}
        for future in tqdm(as_completed(futures), total=len(files), desc="Loading patients"):
            patients[futures[future]] = future.result()

    logger.info(f"Loaded {len(patients)} patients from '{raw_dir}'")
    return patients


# ─── Missing value handling ───────────────────────────────────────────────────

def handle_missing(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
        df_filled  – forward-filled then backward-filled then zero-filled.
        df_mask    – binary mask, 1 = originally observed, 0 = imputed.
    """
    df_mask = (~df[FEATURE_COLS].isna()).astype(np.float32)
    df_filled = df.copy()
    df_filled[FEATURE_COLS] = (
        df_filled[FEATURE_COLS]
        .ffill()
        .bfill()
        .fillna(0.0)
    )
    return df_filled, df_mask


# ─── Windowing ───────────────────────────────────────────────────────────────

def _window_patient(
    args: tuple,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Process a single patient: impute + window. Used for parallel execution."""
    pid, df, window_size, step_size = args
    df_filled, df_mask = handle_missing(df)
    feats = df_filled[FEATURE_COLS].values.astype(np.float32)
    masks = df_mask.values.astype(np.float32)
    labels = df_filled[LABEL_COL].values.astype(np.float32)

    T = len(feats)
    starts = np.arange(0, T - window_size + 1, step_size)
    if len(starts) == 0:
        return None

    # Vectorised sliding window via stride tricks
    idx = starts[:, None] + np.arange(window_size)          # (n_windows, W)
    X_pat = feats[idx]                                       # (n_windows, W, F)
    M_pat = masks[idx]                                       # (n_windows, W, F)
    y_pat = (labels[idx].max(axis=1) > 0).astype(np.float32)  # (n_windows,)
    pid_pat = np.full(len(starts), pid, dtype=np.int32)

    return X_pat, M_pat, y_pat, pid_pat


def create_windows(
    patients: List[pd.DataFrame],
    window_size: int = 24,
    step_size: int = 6,
    n_workers: int = 8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Slide a window over each patient's time series (parallelised).

    Returns:
        X      – (N, window_size, N_FEATURES)  feature windows
        M      – (N, window_size, N_FEATURES)  observation masks
        y      – (N,)                           labels (1 if sepsis onset within window)
        pids   – (N,)                           patient index
    """
    args = [(pid, df, window_size, step_size) for pid, df in enumerate(patients)]

    X_list, M_list, y_list, pid_list = [], [], [], []
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        for result in tqdm(executor.map(_window_patient, args), total=len(args), desc="Windowing"):
            if result is None:
                continue
            X_pat, M_pat, y_pat, pid_pat = result
            X_list.append(X_pat)
            M_list.append(M_pat)
            y_list.append(y_pat)
            pid_list.append(pid_pat)

    if not X_list:
        F = len(FEATURE_COLS)
        return (
            np.empty((0, window_size, F), dtype=np.float32),
            np.empty((0, window_size, F), dtype=np.float32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.int32),
        )

    X = np.concatenate(X_list, axis=0)
    M = np.concatenate(M_list, axis=0)
    y = np.concatenate(y_list, axis=0).astype(np.float32)
    pids = np.concatenate(pid_list, axis=0).astype(np.int32)
    logger.info(
        f"Created {len(X)} windows | positives: {y.sum():.0f} "
        f"({100*y.mean():.2f}%)"
    )
    return X, M, y, pids


# ─── Normalisation ────────────────────────────────────────────────────────────

def fit_scaler(X_train: np.ndarray) -> StandardScaler:
    """Fit a StandardScaler on the training split (reshape to 2-D first)."""
    N, T, F = X_train.shape
    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, F))
    return scaler


def apply_scaler(X: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    N, T, F = X.shape
    return scaler.transform(X.reshape(-1, F)).reshape(N, T, F)


# ─── Train / Val / Test split ─────────────────────────────────────────────────

def split_by_patient(
    X: np.ndarray,
    M: np.ndarray,
    y: np.ndarray,
    pids: np.ndarray,
    val_ratio: float = 0.10,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> dict:
    """
    Split at the *patient* level to avoid data leakage,
    then return a dict with 'train', 'val', 'test' keys.
    """
    rng = np.random.default_rng(seed)
    unique_pids = np.unique(pids)
    rng.shuffle(unique_pids)

    n_test = int(len(unique_pids) * test_ratio)
    n_val = int(len(unique_pids) * val_ratio)

    test_pids = set(unique_pids[:n_test])
    val_pids = set(unique_pids[n_test: n_test + n_val])
    train_pids = set(unique_pids[n_test + n_val:])

    def mask_for(pid_set):
        return np.isin(pids, list(pid_set))

    splits = {}
    for name, pid_set in [("train", train_pids), ("val", val_pids), ("test", test_pids)]:
        m = mask_for(pid_set)
        splits[name] = {"X": X[m], "M": M[m], "y": y[m]}
        logger.info(
            f"  {name}: {m.sum()} windows, "
            f"{splits[name]['y'].sum():.0f} positives "
            f"({100*splits[name]['y'].mean():.2f}%)"
        )
    return splits


# ─── Label ratio sub-sampling (semi-supervised) ───────────────────────────────

def subsample_labels(
    y: np.ndarray,
    label_ratio: float,
    seed: int = 42,
) -> np.ndarray:
    """
    Return a boolean array marking which samples are 'labelled'.
    label_ratio applies to *positive* samples; all negatives are treated as
    unlabelled (their true label is hidden during semi-supervised training).
    """
    rng = np.random.default_rng(seed)
    is_positive = y == 1
    pos_idx = np.where(is_positive)[0]
    n_labelled = max(1, int(len(pos_idx) * label_ratio))
    labelled_pos = rng.choice(pos_idx, size=n_labelled, replace=False)

    labelled_mask = np.zeros(len(y), dtype=bool)
    labelled_mask[labelled_pos] = True
    logger.info(
        f"Semi-supervised: {labelled_mask.sum()} labelled positives "
        f"out of {is_positive.sum()} ({100*label_ratio:.0f}%)"
    )
    return labelled_mask


# ─── Full pipeline entry point ────────────────────────────────────────────────

def run_preprocessing(cfg: dict) -> dict:
    """
    End-to-end preprocessing. Returns a dict with split data + scaler.
    Saves processed arrays to disk for fast reloading.
    """
    processed_dir = Path(cfg["data"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    cache_path = processed_dir / "splits.npz"

    if cache_path.exists():
        logger.info(f"Loading cached preprocessed data from {cache_path}")
        data = dict(np.load(cache_path, allow_pickle=True))
        # numpy savez stores dicts as 0-d object arrays; unwrap
        splits = data["splits"].item()
        scaler = data["scaler"].item()
        return {"splits": splits, "scaler": scaler}

    patients = load_raw_patients(cfg["data"]["raw_dir"])
    X, M, y, pids = create_windows(
        patients,
        window_size=cfg["data"]["window_size"],
        step_size=cfg["data"]["step_size"],
    )
    splits = split_by_patient(
        X, M, y, pids,
        val_ratio=cfg["data"]["val_ratio"],
        test_ratio=cfg["data"]["test_ratio"],
        seed=cfg["data"]["seed"],
    )

    scaler = fit_scaler(splits["train"]["X"])
    for split in splits.values():
        split["X"] = apply_scaler(split["X"], scaler).astype(np.float32)

    # Add labelled mask for semi-supervised training
    splits["train"]["labelled_mask"] = subsample_labels(
        splits["train"]["y"],
        label_ratio=cfg["data"]["label_ratio"],
        seed=cfg["data"]["seed"],
    )

    np.savez(cache_path, splits=splits, scaler=scaler)
    logger.info(f"Saved preprocessed data to {cache_path}")
    return {"splits": splits, "scaler": scaler}
