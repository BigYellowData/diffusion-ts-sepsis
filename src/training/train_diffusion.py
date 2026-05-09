"""
Boucle de pré-entraînement pour le modèle Diffusion-TS.

Stratégie :
  - Entraîner sur TOUTES les fenêtres (étiquetées + non étiquetées) sans information d'étiquette
    → apprentissage non supervisé de la distribution de trajectoire des patients
  - Lorsque des échantillons étiquetés sont dans le lot, entraîner également le chemin conditionnel
    avec le dropout 'classifier-free guidance'
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

logger = logging.getLogger(__name__)


def train_diffusion(
    model,
    dataloader,
    cfg: dict,
    device: torch.device,
    save_dir: str = "checkpoints",
) -> None:
    """
    Entraîne le modèle DiffusionTS.

    Args :
        model      – Instance de DiffusionTS
        dataloader – DataLoader fournissant {'x', 'mask', 'y', 'labelled'}
        cfg        – Dictionnaire de configuration complet
        device     – Périphérique (device) torch
        save_dir   – Répertoire de sauvegarde des points de contrôle (checkpoints)
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    epochs = cfg["training"]["diffusion_epochs"]
    lr = cfg["training"]["diffusion_lr"]

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    model.to(device)
    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in tqdm(dataloader, desc=f"[Diffusion] Epoch {epoch}/{epochs}", leave=False):
            x = batch["x"].to(device)                    # (B, T, F)
            mask = batch["mask"].to(device)              # (B, T, F)
            y = batch["y"].to(device)                    # (B,)
            labelled = batch["labelled"].to(device)      # (B,) – 1 si l'étiquette est connue

            # Construit le conditionnement de classe : les échantillons étiquetés obtiennent leur vraie étiquette,
            # les non étiquetés obtiennent le UNCOND_TOKEN (sera géré dans loss())
            cond = y.long()
            # Masque les échantillons non étiquetés avec le token inconditionnel
            cond[labelled == 0] = model.UNCOND_TOKEN

            optimizer.zero_grad()
            loss = model.loss(x, cond=cond, mask=mask, p_uncond=0.1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(dataloader)
        logger.info(f"[Diffusion] Epoch {epoch:03d} | loss={avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt_path = os.path.join(save_dir, "diffusion_best.pt")
            torch.save(model.state_dict(), ckpt_path)

    logger.info(f"[Diffusion] Training done. Best loss={best_loss:.4f}")
    torch.save(model.state_dict(), os.path.join(save_dir, "diffusion_final.pt"))
