"""
Wrappers PyTorch Dataset pour les fenêtres prétraitées de PhysioNet 2019.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


class SepsisDataset(Dataset):
    """
    Jeu de données pour un seul ensemble (train / val / test).

    Args :
        X              – (N, T, F) fenêtres de caractéristiques normalisées
        M              – (N, T, F) masques d'observation
        y              – (N,)      étiquettes binaires de sepsis
        labelled_mask  – (N,)      bool : quels échantillons ont des étiquettes visibles
                         (None → tous étiquetés, utilisé pour val/test)
    """

    def __init__(
        self,
        X: np.ndarray,
        M: np.ndarray,
        y: np.ndarray,
        labelled_mask: Optional[np.ndarray] = None,
    ):
        self.X = torch.from_numpy(X)                          # (N, T, F)
        self.M = torch.from_numpy(M)                          # (N, T, F)
        self.y = torch.from_numpy(y).float()                  # (N,)
        if labelled_mask is not None:
            self.labelled = torch.from_numpy(labelled_mask.astype(np.float32))
        else:
            self.labelled = torch.ones(len(y))                # tous étiquetés

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> dict:
        return {
            "x": self.X[idx],           # (T, F)
            "mask": self.M[idx],        # (T, F)
            "y": self.y[idx],           # scalaire
            "labelled": self.labelled[idx],  # 1.0 or 0.0
        }


# ─── Fonctions d'aide pour DataLoader ───────────────────────────────────────────────────────

def make_weighted_sampler(y: np.ndarray) -> WeightedRandomSampler:
    """Sur-échantillonne la classe minoritaire (sepsis) pendant l'entraînement."""
    class_counts = np.bincount(y.astype(int))
    weights = 1.0 / class_counts[y.astype(int)]
    return WeightedRandomSampler(
        weights=torch.from_numpy(weights).float(),
        num_samples=len(weights),
        replacement=True,
    )


def build_dataloaders(splits: dict, cfg: dict) -> dict:
    """
    Construit les DataLoaders pour l'entraînement / validation / test.
    L'entraînement utilise un WeightedRandomSampler pour gérer le déséquilibre des classes.
    """
    train_split = splits["train"]
    train_ds = SepsisDataset(
        train_split["X"],
        train_split["M"],
        train_split["y"],
        labelled_mask=train_split.get("labelled_mask"),
    )
    val_ds = SepsisDataset(splits["val"]["X"], splits["val"]["M"], splits["val"]["y"])
    test_ds = SepsisDataset(splits["test"]["X"], splits["test"]["M"], splits["test"]["y"])

    sampler = make_weighted_sampler(train_split["y"])
    batch = cfg["training"]["classifier_batch_size"]

    loaders = {
        "train": DataLoader(
            train_ds,
            batch_size=batch,
            sampler=sampler,
            num_workers=0,
            pin_memory=True,
            drop_last=True,
        ),
        "val": DataLoader(
            val_ds,
            batch_size=batch * 2,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        ),
        "test": DataLoader(
            test_ds,
            batch_size=batch * 2,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        ),
    }
    return loaders


def build_diffusion_loader(splits: dict, cfg: dict) -> DataLoader:
    """
    DataLoader pour le pré-entraînement non supervisé par diffusion.
    Utilise TOUTES les fenêtres (étiquetées + non étiquetées) sans information d'étiquette.
    """
    train_split = splits["train"]
    ds = SepsisDataset(
        train_split["X"],
        train_split["M"],
        train_split["y"],
    )
    return DataLoader(
        ds,
        batch_size=cfg["training"]["diffusion_batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
