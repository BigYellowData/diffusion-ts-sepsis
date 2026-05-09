"""
Baseline — TimeGAN semi-supervisé (Yoon et al., NeurIPS 2019).

TimeGAN entraîne conjointement 4 réseaux (plus un superviseur) :
  - Embedder E      : espace des caractéristiques → espace latent
  - Recovery R      : espace latent → espace des caractéristiques
  - Generator G     : bruit → espace latent (synthétique)
  - Discriminator D : réel vs synthétique dans l'espace latent
  - Supervisor S    : impose la cohérence par étapes dans l'espace latent

Configuration semi-supervisée : après un pré-entraînement non supervisé, nous générons
des échantillons synthétiques de sepsis et entraînons un classificateur BiLSTM sur l'union
des données réelles étiquetées et des échantillons synthétiques.
"""

from __future__ import annotations

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from ..utils.metrics import compute_metrics, optimal_threshold

logger = logging.getLogger(__name__)


# ─── TimeGAN components ───────────────────────────────────────────────────────

class GRUNet(nn.Module):
    """Bloc GRU générique utilisé comme Embedder, Recovery, Generator et Discriminator."""
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, n_layers: int = 3):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden_dim, n_layers, batch_first=True)
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.head(out)


class TimeGAN(nn.Module):
    def __init__(self, n_features: int, hidden_dim: int = 64, n_layers: int = 3):
        super().__init__()
        self.embedder     = GRUNet(n_features,  hidden_dim, hidden_dim, n_layers)
        self.recovery     = GRUNet(hidden_dim,  hidden_dim, n_features, n_layers)
        self.generator    = GRUNet(n_features,  hidden_dim, hidden_dim, n_layers)
        self.discriminator = GRUNet(hidden_dim, hidden_dim, 1,          n_layers)
        self.supervisor   = GRUNet(hidden_dim,  hidden_dim, hidden_dim, n_layers)

    def embed(self, x):      return torch.sigmoid(self.embedder(x))
    def recover(self, h):    return self.recovery(h)
    def generate(self, z):   return torch.sigmoid(self.generator(z))
    def supervise(self, h):  return torch.sigmoid(self.supervisor(h))
    def discriminate(self, h): return self.discriminator(h)


# ─── TimeGAN training ─────────────────────────────────────────────────────────

def _random_noise(B: int, T: int, F: int, device) -> torch.Tensor:
    """Échantillonne un bruit gaussien standard de dimension (B, T, F)."""
    return torch.randn(B, T, F, device=device)


def train_timegan(
    splits: dict,
    cfg: dict,
    device: torch.device,
    hidden_dim: int = 64,
    n_layers: int = 3,
    pretrain_epochs: int = 50,
    joint_epochs: int = 50,
    batch_size: int = 128,
) -> dict:
    """Entraîne TimeGAN de bout en bout et évalue un classificateur BiLSTM en aval.

    Pipeline :
      1. Phase 1 — Pré-entraînement de l'Embedder et du Recovery sur la reconstruction.
      2. Phase 2 — Pré-entraînement du Supervisor sur la prédiction de l'étape suivante dans l'espace latent.
      3. Phase 3 — Entraînement antagoniste conjoint (mises à jour de G, E, D).
      4. Génération d'échantillons synthétiques de sepsis.
      5. Entraînement d'un classificateur BiLSTM sur (données réelles étiquetées + synthétiques).
      6. Calibrage du seuil de décision via l'indice de Youden sur l'ensemble de validation.

    Retours
    -------
    dict
        ``{"val": metrics_dict, "test": metrics_dict}`` où chaque dictionnaire de métriques
        contient AUROC, AUPRC, F1, ECE et l'utilité PhysioNet.
    """
    train = splits["train"]
    X_all = torch.from_numpy(train["X"]).to(device)      # (N, T, F)
    y_all = torch.from_numpy(train["y"]).to(device)

    B_total, T, F = X_all.shape
    model = TimeGAN(F, hidden_dim, n_layers).to(device)

    opt_E = optim.Adam(list(model.embedder.parameters()) +
                       list(model.recovery.parameters()), lr=1e-3)
    opt_G = optim.Adam(list(model.generator.parameters()) +
                       list(model.supervisor.parameters()), lr=1e-3)
    opt_D = optim.Adam(model.discriminator.parameters(), lr=1e-3)

    ds = TensorDataset(X_all)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    # ── Phase 1: pré-entraînement Embedder + Recovery sur la reconstruction ──────────────
    logger.info(f"[TimeGAN] Phase 1 — Embedder/Recovery pre-training ({pretrain_epochs} epochs)")
    for epoch in range(1, pretrain_epochs + 1):
        for (xb,) in loader:
            h = model.embed(xb)
            x_hat = model.recover(h)
            loss_r = nn.MSELoss()(x_hat, xb)
            opt_E.zero_grad(); loss_r.backward(); opt_E.step()

    # ── Phase 2: pré-entraînement Supervisor (prédiction étape suivante espace latent) ──
    logger.info(f"[TimeGAN] Phase 2 — Supervisor pre-training ({pretrain_epochs} epochs)")
    opt_S = optim.Adam(model.supervisor.parameters(), lr=1e-3)
    for epoch in range(1, pretrain_epochs + 1):
        for (xb,) in loader:
            h = model.embed(xb).detach()
            h_hat = model.supervise(h[:, :-1, :])
            loss_s = nn.MSELoss()(h_hat, h[:, 1:, :])
            opt_S.zero_grad(); loss_s.backward(); opt_S.step()

    # ── Phase 3: entraînement antagoniste conjoint ──────────────────────────────────
    logger.info(f"[TimeGAN] Phase 3 — Joint training ({joint_epochs} epochs)")
    bce = nn.BCEWithLogitsLoss()
    for epoch in range(1, joint_epochs + 1):
        for (xb,) in loader:
            Bx = xb.shape[0]
            z = _random_noise(Bx, T, F, device)

            # Étape du Générateur
            e_hat = model.generate(z)
            h_hat = model.supervise(e_hat)
            y_fake = model.discriminate(h_hat)
            h_real = model.embed(xb).detach()
            loss_g = (bce(y_fake, torch.ones_like(y_fake))
                      + 100 * nn.MSELoss()(h_hat[:, 1:], h_real[:, 1:])
                      + 100 * (h_hat.std(1).mean() - h_real.std(1).mean()).abs())
            opt_G.zero_grad(); loss_g.backward(); opt_G.step()

            # Étape de l'Embedder
            h = model.embed(xb)
            x_hat = model.recover(h)
            h_sup = model.supervise(h[:, :-1, :])
            loss_e = (nn.MSELoss()(x_hat, xb)
                      + 0.1 * nn.MSELoss()(h_sup, h[:, 1:, :].detach()))
            opt_E.zero_grad(); loss_e.backward(); opt_E.step()

            # Étape du Discriminateur
            z = _random_noise(Bx, T, F, device)
            e_hat = model.generate(z).detach()
            h_hat = model.supervise(e_hat).detach()
            h_real = model.embed(xb).detach()
            y_real = model.discriminate(h_real)
            y_fake = model.discriminate(h_hat)
            loss_d = (bce(y_real, torch.ones_like(y_real))
                      + bce(y_fake, torch.zeros_like(y_fake)))
            if loss_d.item() > 0.15:   # n'entraîne D que s'il est trop faible
                opt_D.zero_grad(); loss_d.backward(); opt_D.step()

    # ── Génération d'échantillons synthétiques de sepsis ────────────────────────────────────
    n_pos_real = int((y_all == 1).sum().item())
    n_synth = n_pos_real * cfg["generation"]["n_synthetic_per_real"]
    logger.info(f"[TimeGAN] Generating {n_synth} synthetic sepsis samples")

    model.eval()
    with torch.no_grad():
        z = _random_noise(n_synth, T, F, device)
        e_hat = model.generate(z)
        x_synth = model.recover(e_hat).cpu().numpy().astype(np.float32)

    # ── Classificateur BiLSTM sur (réel étiqueté + synthétique) ─────────────────────
    labelled = train["labelled_mask"]
    keep = labelled | (train["y"] == 0)
    X_real = train["X"][keep].astype(np.float32)
    y_real = train["y"][keep].astype(np.float32)
    y_synth = np.ones(n_synth, dtype=np.float32)

    X_tr = np.concatenate([X_real, x_synth], axis=0)
    y_tr = np.concatenate([y_real, y_synth], axis=0)

    n_pos = int(y_tr.sum())
    n_neg = int((y_tr == 0).sum())
    logger.info(f"[TimeGAN] Classifier training on {len(X_tr)} samples | pos={n_pos} neg={n_neg}")

    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)
    clf_ds = TensorDataset(
        torch.from_numpy(X_tr).to(device),
        torch.from_numpy(y_tr).to(device),
    )
    clf_loader = DataLoader(clf_ds, batch_size=128, shuffle=True, drop_last=True)

    clf = nn.Sequential(
        nn.GRU(F, 128, 2, batch_first=True, bidirectional=True),
    )

    class BiLSTMClf(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(F, 128, 2, batch_first=True,
                              bidirectional=True, dropout=0.3)
            self.head = nn.Linear(256, 1)
        def forward(self, x):
            out, _ = self.gru(x)
            return self.head(out[:, -1, :]).squeeze(-1)

    clf = BiLSTMClf().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt_clf = optim.AdamW(clf.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt_clf, T_max=30)

    best_auroc, best_state, patience, no_improve = 0.0, None, 10, 0
    for epoch in range(1, 51):
        clf.train()
        for xb, yb in clf_loader:
            opt_clf.zero_grad()
            nn.utils.clip_grad_norm_(clf.parameters(), 1.0)
            criterion(clf(xb), yb).backward()
            opt_clf.step()
        sched.step()

        val_prob = _clf_predict(clf, splits["val"], device)
        val_auroc = compute_metrics(splits["val"]["y"], val_prob)["auroc"]
        if val_auroc > best_auroc:
            best_auroc = val_auroc
            best_state = {k: v.cpu().clone() for k, v in clf.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info(f"[TimeGAN] Early stopping at epoch {epoch}")
                break

    clf.load_state_dict(best_state)
    val_prob = _clf_predict(clf, splits["val"], device)
    threshold = optimal_threshold(splits["val"]["y"], val_prob)

    results = {}
    for split_name in ("val", "test"):
        prob = _clf_predict(clf, splits[split_name], device)
        results[split_name] = compute_metrics(splits[split_name]["y"], prob, threshold)
        logger.info(
            f"[TimeGAN] {split_name} | AUROC={results[split_name]['auroc']:.4f} | "
            f"AUPRC={results[split_name]['auprc']:.4f} | F1={results[split_name]['f1']:.4f}"
        )
    return results


@torch.no_grad()
def _clf_predict(clf, split: dict, device: torch.device) -> np.ndarray:
    """Exécute le classificateur en mode évaluation et retourne les probabilités sigmoïdes."""
    clf.eval()
    X = torch.from_numpy(split["X"]).to(device)
    ds = TensorDataset(X)
    loader = DataLoader(ds, batch_size=512)
    probs = []
    for (xb,) in loader:
        probs.append(torch.sigmoid(clf(xb)).cpu().numpy())
    return np.concatenate(probs)
