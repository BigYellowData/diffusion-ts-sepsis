"""
Baseline 2 — Bidirectional LSTM on labelled data only.

Trained exclusively on labelled samples (no unlabelled, no augmentation).
No MC Dropout — single deterministic forward pass.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from ..utils.metrics import compute_metrics, optimal_threshold

logger = logging.getLogger(__name__)


# ─── Model ────────────────────────────────────────────────────────────────────

class LSTMClassifier(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 128, n_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size * 2, 1)   # *2 for bidirectional

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)                        # (B, T, 2H)
        last = out[:, -1, :]                         # (B, 2H)  last step
        return self.head(self.drop(last)).squeeze(-1)


# ─── Training ─────────────────────────────────────────────────────────────────

def train_lstm(splits: dict, cfg: dict, device: torch.device) -> dict:
    """
    Train BiLSTM using only labelled samples.
    Returns val and test metrics.
    """
    train = splits["train"]
    labelled = train["labelled_mask"]
    keep = labelled | (train["y"] == 0)

    X_tr = torch.from_numpy(train["X"][keep])
    M_tr = torch.from_numpy(train["M"][keep])
    y_tr = torch.from_numpy(train["y"][keep])

    # Concatenate mask as extra features (same as classifier)
    X_tr = torch.cat([X_tr, M_tr], dim=-1)

    n_pos = int(y_tr.sum())
    n_neg = int((y_tr == 0).sum())
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)
    logger.info(f"[LSTM] Training on {len(X_tr)} samples | pos={n_pos} neg={n_neg}")

    ds = TensorDataset(X_tr, y_tr)
    loader = DataLoader(ds, batch_size=128, shuffle=True, drop_last=True)

    n_features = X_tr.shape[-1]
    model = LSTMClassifier(n_features=n_features).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

    best_val_auroc = 0.0
    best_state = None
    patience, no_improve = 10, 0

    for epoch in range(1, 51):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        val_metrics = _eval_lstm(model, splits["val"], device)
        if val_metrics["auroc"] > best_val_auroc:
            best_val_auroc = val_metrics["auroc"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info(f"[LSTM] Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)

    # Calibrate threshold on val, apply to test
    val_prob = _predict_proba(model, splits["val"], device)
    threshold = optimal_threshold(splits["val"]["y"], val_prob)

    results = {}
    for split_name in ("val", "test"):
        prob = _predict_proba(model, splits[split_name], device)
        results[split_name] = compute_metrics(splits[split_name]["y"], prob, threshold)
        logger.info(
            f"[LSTM] {split_name} | AUROC={results[split_name]['auroc']:.4f} | "
            f"AUPRC={results[split_name]['auprc']:.4f} | F1={results[split_name]['f1']:.4f}"
        )
    return results


@torch.no_grad()
def _predict_proba(model: LSTMClassifier, split: dict, device: torch.device) -> np.ndarray:
    model.eval()
    X = torch.cat([
        torch.from_numpy(split["X"]),
        torch.from_numpy(split["M"]),
    ], dim=-1)
    ds = TensorDataset(X)
    loader = DataLoader(ds, batch_size=512)
    probs = []
    for (xb,) in loader:
        logit = model(xb.to(device))
        probs.append(torch.sigmoid(logit).cpu().numpy())
    return np.concatenate(probs)


@torch.no_grad()
def _eval_lstm(model: LSTMClassifier, split: dict, device: torch.device) -> dict:
    prob = _predict_proba(model, split, device)
    return compute_metrics(split["y"], prob)
