"""
Boucle d'entraînement pour le classificateur Transformer avec MC Dropout.

Stratégie :
  1. (Optionnel) Augmenter l'ensemble d'entraînement avec des échantillons de sepsis synthétiques
     générés par le modèle Diffusion-TS pré-entraîné.
  2. Entraîner avec une perte BCE pondérée par classe.
  3. Valider avec AUROC / AUPRC / Score d'utilité PhysioNet.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, ConcatDataset, TensorDataset
from tqdm import tqdm

from ..models.classifier import SepsisClassifier
from ..models.diffusion_ts import DiffusionTS
from ..utils.metrics import compute_metrics

logger = logging.getLogger(__name__)


# ─── Perte pondérée par classe ──────────────────────────────────────────────────────

def effective_num_weights(y: np.ndarray, beta: float = 0.9999) -> torch.Tensor:
    """
    Poids de perte équilibrés par classe (Cui et al., 2019).
    Plus robuste que la fréquence inverse naïve pour un déséquilibre extrême.
    """
    counts = np.bincount(y.astype(int), minlength=2).astype(float)
    en = (1 - beta ** counts) / (1 - beta)
    weights = 1.0 / en
    weights = weights / weights.sum() * 2   # normalise pour que la moyenne ≈ 1
    return torch.tensor(weights, dtype=torch.float32)


# ─── Augmentation synthétique ───────────────────────────────────────────────────

def augment_with_diffusion(
    diffusion_model: DiffusionTS,
    train_loader: DataLoader,
    cfg: dict,
    device: torch.device,
) -> DataLoader:
    """
    Génère des échantillons synthétiques de sepsis et retourne un DataLoader augmenté.
    """
    gen_cfg = cfg["generation"]
    # Compte les vrais positifs directement depuis le jeu de données (et non l'échantillonneur)
    n_real_pos = int(train_loader.dataset.y.sum().item())
    n_synthetic = int(n_real_pos * gen_cfg["n_synthetic_per_real"])
    logger.info(f"[Augment] Generating {n_synthetic} synthetic sepsis samples…")

    diffusion_model.eval()
    synth_x = diffusion_model.generate_class(
        label=1,
        n_samples=n_synthetic,
        device=device,
        fast=True,
        n_ddim_steps=50,
    )  # (n_synthetic, T, F)

    synth_y = torch.ones(n_synthetic, dtype=torch.float32)
    synth_mask = torch.ones_like(synth_x)       # synthétique → masque entièrement observé
    synth_labelled = torch.ones(n_synthetic)

    # Enveloppe dans un TensorDataset compatible avec le format SepsisDataset
    from ..data.dataset import SepsisDataset
    import numpy as np
    synth_ds = SepsisDataset(
        synth_x.cpu().numpy(),
        synth_mask.cpu().numpy(),
        synth_y.numpy(),
    )

    # Fusionne avec le jeu de données d'entraînement original
    combined = ConcatDataset([train_loader.dataset, synth_ds])
    new_loader = DataLoader(
        combined,
        batch_size=train_loader.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
    logger.info(f"[Augment] Combined dataset size: {len(combined)}")
    return new_loader


# ─── Boucle d'entraînement principale ───────────────────────────────────────────────────────

def train_classifier(
    classifier: SepsisClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: dict,
    device: torch.device,
    diffusion_model: Optional[DiffusionTS] = None,
    save_dir: str = "checkpoints",
) -> SepsisClassifier:
    """
    Entraîne le SepsisClassifier.

    Args :
        classifier      – Instance de SepsisClassifier
        train_loader    – DataLoader pour l'ensemble d'entraînement
        val_loader      – DataLoader pour l'ensemble de validation
        cfg             – Dictionnaire de configuration complet
        device          – Périphérique (device) torch
        diffusion_model – DiffusionTS pré-entraîné (utilisé pour l'augmentation)
        save_dir        – Répertoire de sauvegarde des points de contrôle (checkpoints)
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Augmentation de données optionnelle
    if diffusion_model is not None:
        train_loader = augment_with_diffusion(
            diffusion_model, train_loader, cfg, device
        )

    # Poids des classes à partir des étiquettes d'entraînement
    all_y = np.array([
        batch["y"].numpy() for batch in train_loader
    ], dtype=object)
    all_y = np.concatenate([a for a in all_y]).astype(int)
    class_weights = effective_num_weights(all_y, beta=cfg["training"]["class_weight_beta"])
    pos_weight = torch.tensor([class_weights[1] / class_weights[0]], device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    epochs = cfg["training"]["classifier_epochs"]
    lr = cfg["training"]["classifier_lr"]
    optimizer = optim.AdamW(classifier.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    classifier.to(device)
    best_auroc = 0.0
    best_ckpt = os.path.join(save_dir, "classifier_best.pt")
    patience = cfg["training"].get("early_stopping_patience", 10)
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        # ── Entraînement ──────────────────────────────────────────────────────────
        classifier.train()
        total_loss = 0.0

        for batch in tqdm(train_loader, desc=f"[Classifier] Epoch {epoch}/{epochs}", leave=False):
            x = batch["x"].to(device)
            mask = batch["mask"].to(device)
            y = batch["y"].to(device)

            optimizer.zero_grad()
            logit = classifier(x, mask)
            loss = criterion(logit, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(train_loader)

        # ── Validation ────────────────────────────────────────────────────────
        metrics = evaluate_classifier(classifier, val_loader, device, cfg)
        logger.info(
            f"[Classifier] Epoch {epoch:03d} | loss={avg_loss:.4f} | "
            f"AUROC={metrics['auroc']:.4f} | AUPRC={metrics['auprc']:.4f} | "
            f"F1={metrics['f1']:.4f}"
        )

        if metrics["auroc"] > best_auroc:
            best_auroc = metrics["auroc"]
            epochs_no_improve = 0
            torch.save(classifier.state_dict(), best_ckpt)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(f"[Classifier] Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    logger.info(f"[Classifier] Training done. Best AUROC={best_auroc:.4f}")
    # Charge les meilleurs poids
    classifier.load_state_dict(torch.load(best_ckpt, map_location=device))
    torch.save(classifier.state_dict(), os.path.join(save_dir, "classifier_final.pt"))
    return classifier


# ─── Fonction d'aide à l'évaluation ────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_classifier(
    classifier: SepsisClassifier,
    loader: DataLoader,
    device: torch.device,
    cfg: dict,
) -> dict:
    """Exécute l'évaluation en mode déterministe (dropout désactivé)."""
    classifier.eval()
    all_probs, all_labels = [], []

    for batch in loader:
        x = batch["x"].to(device)
        mask = batch["mask"].to(device)
        y = batch["y"]
        logit = classifier(x, mask)
        prob = torch.sigmoid(logit).cpu()
        all_probs.append(prob)
        all_labels.append(y)

    probs = torch.cat(all_probs).numpy()
    labels = torch.cat(all_labels).numpy()
    return compute_metrics(labels, probs)
