"""
Modèle de base 3 — Diffusion-TS classique + classificateur (pas de semi-supervisé, pas d'augmentation).

Utilise la même architecture de classificateur Transformer mais :
  - Entraîné uniquement sur les données étiquetées (aucune augmentation synthétique)
  - Pas de MC Dropout lors de l'inférence (passage déterministe unique)
  - Seuil calibré sur l'ensemble de validation
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from ..models.classifier import SepsisClassifier
from ..utils.metrics import compute_metrics, optimal_threshold

logger = logging.getLogger(__name__)


def train_vanilla_classifier(splits: dict, cfg: dict, device: torch.device) -> dict:
    """
    Entraîne le classificateur Transformer uniquement sur les données étiquetées, sans augmentation.
    """
    train = splits["train"]
    labelled = train["labelled_mask"]
    keep = labelled | (train["y"] == 0)

    X_tr = torch.from_numpy(train["X"][keep])
    M_tr = torch.from_numpy(train["M"][keep])
    y_tr = torch.from_numpy(train["y"][keep])

    n_pos = int(y_tr.sum())
    n_neg = int((y_tr == 0).sum())
    logger.info(f"[Vanilla] Training on {len(X_tr)} samples | pos={n_pos} neg={n_neg}")

    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)
    ds = TensorDataset(X_tr, M_tr, y_tr)
    loader = DataLoader(ds, batch_size=128, shuffle=True, drop_last=True)

    clf_cfg = cfg["classifier"]
    diff_cfg = cfg["diffusion"]
    model = SepsisClassifier(
        n_features=diff_cfg["n_features"],
        seq_len=diff_cfg["seq_len"],
        d_model=clf_cfg["d_model"],
        n_heads=clf_cfg["n_heads"],
        n_layers=clf_cfg["n_layers"],
        d_ff=clf_cfg["d_ff"],
        dropout=clf_cfg["dropout"],
    ).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    best_auroc = 0.0
    best_state = None
    patience, no_improve = 10, 0

    for epoch in range(1, 101):
        model.train()
        for xb, mb, yb in loader:
            xb, mb, yb = xb.to(device), mb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb, mb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        val_metrics = _eval(model, splits["val"], device)
        if val_metrics["auroc"] > best_auroc:
            best_auroc = val_metrics["auroc"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info(f"[Vanilla] Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)

    val_prob = _predict(model, splits["val"], device)
    threshold = optimal_threshold(splits["val"]["y"], val_prob)

    results = {}
    for split_name in ("val", "test"):
        prob = _predict(model, splits[split_name], device)
        results[split_name] = compute_metrics(splits[split_name]["y"], prob, threshold)
        logger.info(
            f"[Vanilla] {split_name} | AUROC={results[split_name]['auroc']:.4f} | "
            f"AUPRC={results[split_name]['auprc']:.4f} | F1={results[split_name]['f1']:.4f}"
        )
    return results


@torch.no_grad()
def _predict(model: SepsisClassifier, split: dict, device: torch.device) -> np.ndarray:
    """Effectue une prédiction sur l'ensemble de données et retourne les probabilités."""
    model.eval()
    X = torch.from_numpy(split["X"])
    M = torch.from_numpy(split["M"])
    ds = TensorDataset(X, M)
    loader = DataLoader(ds, batch_size=512)
    probs = []
    for xb, mb in loader:
        logit = model(xb.to(device), mb.to(device))
        probs.append(torch.sigmoid(logit).cpu().numpy())
    return np.concatenate(probs)


@torch.no_grad()
def _eval(model: SepsisClassifier, split: dict, device: torch.device) -> dict:
    """Évalue le modèle sur l'ensemble de données."""
    return compute_metrics(split["y"], _predict(model, split, device))
