"""
Génération des figures pour le rapport.

Figures produites (sauvegardées dans figures/) :
  1. roc_curve.pdf         — Courbe ROC avec AUROC annoté
  2. pr_curve.pdf          — Courbe Précision-Rappel avec AUPRC annoté
  3. calibration.pdf       — Diagramme de fiabilité (calibration) + ECE
  4. abstention.pdf        — Courbe d'abstention (précision vs couverture)
  5. uncertainty_dist.pdf  — Distribution de l'incertitude MC Dropout
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import roc_curve, precision_recall_curve

logger = logging.getLogger(__name__)

# ── Style global ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        11,
    "axes.titlesize":   12,
    "axes.labelsize":   11,
    "legend.fontsize":  10,
    "figure.dpi":       150,
    "axes.spines.top":  False,
    "axes.spines.right": False,
})

BLUE   = "#2563EB"
ORANGE = "#EA580C"
GRAY   = "#94A3B8"
GREEN  = "#16A34A"

FIGURES_DIR = Path("figures")


def _save(fig: plt.Figure, name: str) -> None:
    """Sauvegarde la figure dans le dossier des figures."""
    FIGURES_DIR.mkdir(exist_ok=True)
    path = FIGURES_DIR / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"[Plots] Saved {path}")


# ── 1. Courbe ROC ─────────────────────────────────────────────────────────────

def plot_roc(y_true: np.ndarray, y_prob: np.ndarray, auroc: float) -> None:
    """Génère et sauvegarde la courbe ROC."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot(fpr, tpr, color=BLUE, lw=2,
            label=f"Diffusion-TS + MC Dropout\n(AUROC = {auroc:.4f})")
    ax.plot([0, 1], [0, 1], color=GRAY, lw=1, linestyle="--", label="Aléatoire")

    ax.set_xlabel("Taux de faux positifs")
    ax.set_ylabel("Taux de vrais positifs")
    ax.set_title("Courbe ROC")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)

    _save(fig, "roc_curve.pdf")


# ── 2. Courbe Précision-Rappel ────────────────────────────────────────────────

def plot_pr(y_true: np.ndarray, y_prob: np.ndarray, auprc: float,
            threshold: float) -> None:
    """Génère et sauvegarde la courbe Précision-Rappel."""
    prec, rec, thresholds = precision_recall_curve(y_true, y_prob)
    baseline = y_true.mean()

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot(rec, prec, color=ORANGE, lw=2,
            label=f"Diffusion-TS + MC Dropout\n(AUPRC = {auprc:.4f})")
    ax.axhline(baseline, color=GRAY, lw=1, linestyle="--",
               label=f"Aléatoire ({baseline:.3f})")

    # Point correspondant au seuil optimal
    diffs = np.abs(thresholds - threshold)
    idx = int(np.argmin(diffs))
    ax.scatter(rec[idx], prec[idx], color=ORANGE, s=60, zorder=5,
               label=f"Seuil optimal ({threshold:.3f})")

    ax.set_xlabel("Rappel")
    ax.set_ylabel("Précision")
    ax.set_title("Courbe Précision-Rappel")
    ax.legend(loc="upper right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)

    _save(fig, "pr_curve.pdf")


# ── 3. Diagramme de calibration ───────────────────────────────────────────────

def plot_calibration(y_true: np.ndarray, y_prob: np.ndarray,
                     ece: float, n_bins: int = 10) -> None:
    """Génère et sauvegarde le diagramme de calibration (fiabilité) et l'histogramme des probabilités."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers, accs, confs, counts = [], [], [], []

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        bin_centers.append((lo + hi) / 2)
        accs.append(float(y_true[mask].mean()))
        confs.append(float(y_prob[mask].mean()))
        counts.append(int(mask.sum()))

    bin_centers = np.array(bin_centers)
    accs        = np.array(accs)
    confs       = np.array(confs)
    counts      = np.array(counts)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))

    # Diagramme de fiabilité
    ax1.plot([0, 1], [0, 1], color=GRAY, lw=1, linestyle="--",
             label="Calibration parfaite")
    ax1.bar(bin_centers, accs, width=0.08, alpha=0.7, color=BLUE,
            label="Fréquence observée")
    ax1.plot(confs, accs, "o-", color=ORANGE, lw=2, ms=5,
             label="Notre modèle")
    ax1.set_xlabel("Confiance moyenne")
    ax1.set_ylabel("Fréquence observée")
    ax1.set_title(f"Diagramme de fiabilité (ECE = {ece:.4f})")
    ax1.legend(fontsize=9)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    # Histogramme des probabilités prédites
    ax2.hist(y_prob[y_true == 0], bins=50, alpha=0.6, color=GRAY,
             label="Non-sepsis", density=True)
    ax2.hist(y_prob[y_true == 1], bins=50, alpha=0.7, color=BLUE,
             label="Sepsis", density=True)
    ax2.set_xlabel("Probabilité prédite")
    ax2.set_ylabel("Densité")
    ax2.set_title("Distribution des probabilités")
    ax2.legend()

    fig.tight_layout()
    _save(fig, "calibration.pdf")


# ── 4. Courbe d'abstention ────────────────────────────────────────────────────

def plot_abstention(y_true: np.ndarray, y_prob: np.ndarray,
                    uncertainty: np.ndarray, n_points: int = 30,
                    threshold: float = 0.5) -> None:
    """
    Courbe de précision sélective : parmi les prédictions positives (y_pred=1 au
    seuil opérationnel), s'abstenir sur les plus incertaines — en les signalant
    pour examen humain plutôt que de déclencher une alarme automatique — et mesurer
    la précision sur les alertes conservées.

    Cela correspond au cas d'utilisation clinique : la plupart des prédictions sont non-sepsis, les
    alertes sont rares et à forts enjeux. Nous voulons une grande précision parmi les alertes.

    Forme attendue : décroissante de façon monotone de gauche (couverture élevée des alertes
    conservées = précision de base) à droite (couverture faible = précision la plus élevée après
    avoir écarté les alertes les plus incertaines).
    """
    pos_pred = y_prob >= threshold
    yt_pos = y_true[pos_pred].astype(float)   # 1 si vrai sepsis, 0 si FP
    unc_pos = uncertainty[pos_pred]
    n_pos_pred = int(pos_pred.sum())
    baseline_prec = float(yt_pos.mean())

    # Trie les alertes par incertitude (les plus certaines d'abord)
    order = np.argsort(unc_pos)

    coverages, precisions = [], []
    for k in np.linspace(0.05, 1.0, n_points):
        n_keep = max(20, int(k * n_pos_pred))
        idx = order[:n_keep]
        coverages.append(k)
        precisions.append(float(yt_pos[idx].mean()))

    coverages  = np.array(coverages)
    precisions = np.array(precisions)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(coverages * 100, precisions * 100, color=GREEN, lw=2,
            label="Diffusion-TS + MC Dropout")
    ax.axhline(baseline_prec * 100, color=GRAY, lw=1, linestyle="--",
               label=f"Précision sans abstention ({baseline_prec*100:.1f} %)")

    ax.set_xlabel(f"Couverture des alertes (n = {n_pos_pred} prédictions positives)")
    ax.set_ylabel("Précision parmi les alertes conservées (%)")
    ax.set_title("Courbe d'abstention sur les alertes\n"
                 "(les alertes les plus incertaines sont écartées en premier)")
    ax.legend(loc="upper right")
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%d%%"))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%d%%"))

    _save(fig, "abstention.pdf")


# ── 5. Distribution de l'incertitude ─────────────────────────────────────────

def plot_uncertainty_distribution(y_true: np.ndarray, y_prob: np.ndarray,
                                   uncertainty: np.ndarray,
                                   threshold: float) -> None:
    """Génère et sauvegarde l'histogramme de distribution de l'incertitude."""
    y_pred = (y_prob >= threshold).astype(int)
    correct = y_pred == y_true
    incorrect = ~correct

    u_correct   = uncertainty[correct]
    u_incorrect = uncertainty[incorrect]

    ratio = u_incorrect.mean() / (u_correct.mean() + 1e-12)

    fig, ax = plt.subplots(figsize=(5, 4))

    bins = np.linspace(0, uncertainty.max(), 40)
    ax.hist(u_correct,   bins=bins, alpha=0.6, color=GREEN,
            label=f"Prédictions correctes (n={correct.sum():,})", density=True)
    ax.hist(u_incorrect, bins=bins, alpha=0.7, color=ORANGE,
            label=f"Prédictions incorrectes (n={incorrect.sum():,})", density=True)

    ax.axvline(u_correct.mean(),   color=GREEN,  lw=1.5, linestyle="--")
    ax.axvline(u_incorrect.mean(), color=ORANGE, lw=1.5, linestyle="--")

    ax.set_xlabel("Incertitude (variance MC Dropout)")
    ax.set_ylabel("Densité")
    ax.set_title(f"Incertitude : erreurs vs corrects\n(ratio = ×{ratio:.1f})")
    ax.legend(fontsize=9)

    _save(fig, "uncertainty_dist.pdf")


# ── Fonction principale ───────────────────────────────────────────────────────

def generate_all_plots(results_dir: str = "results") -> None:
    """Charge les résultats sauvegardés et génère toutes les figures."""
    results_path = Path(results_dir)

    y_true      = np.load(results_path / "y_true.npy")
    mean_prob   = np.load(results_path / "mean_prob.npy")
    uncertainty = np.load(results_path / "uncertainty.npy")

    with open(results_path / "metrics.json") as f:
        metrics = json.load(f)

    auroc     = metrics["auroc"]
    auprc     = metrics["auprc"]
    ece       = metrics["ece"]
    threshold = metrics["threshold"]

    logger.info(f"[Plots] Generating figures for {len(y_true):,} samples "
                f"(pos={int(y_true.sum())}, threshold={threshold:.4f})")

    plot_roc(y_true, mean_prob, auroc)
    plot_pr(y_true, mean_prob, auprc, threshold)
    plot_calibration(y_true, mean_prob, ece)
    plot_uncertainty_distribution(y_true, mean_prob, uncertainty, threshold)
    # Note: plot_abstention() est conservée comme utilitaire mais non appelée
    # ici. Sur ce dataset, le score MC Dropout ne discrimine pas TP/FP parmi
    # les alertes (cf. discussion §6.3), la courbe est donc peu informative.

    logger.info(f"[Plots] All figures saved to {FIGURES_DIR.resolve()}/")
