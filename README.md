# Diffusion-TS Sepsis — Semi-Supervised Clinical Time Series Classification

Adaptation de [Diffusion-TS](https://github.com/Y-debug-sys/Diffusion-TS) (ICLR 2024) pour la **prédiction précoce du sepsis** sur le dataset PhysioNet/CinC Challenge 2019.

Trois contributions principales :
- **Génération conditionnelle** par diffusion pour rééquilibrer le dataset (~2.3% positifs)
- **Cadre semi-supervisé** : entraînement sur toutes les données, guidage conditionnel sur les labellés uniquement
- **Incertitude bayésienne** via Monte Carlo Dropout — le modèle sait quand il hésite

---

## Résultats (test set, 10% de labels)

| Méthode | AUROC | AUPRC | F1 | ECE |
|---|---|---|---|---|
| XGBoost (labellé seulement) | 0.7835 | 0.0997 | 0.1156 | — |
| BiLSTM (labellé seulement) | 0.8406 | 0.1113 | 0.1326 | — |
| Transformer (sans augmentation) | 0.7881 | 0.0743 | 0.0919 | — |
| **Diffusion-TS + Aug + MC Dropout** | **0.8684** | **0.1356** | 0.1228 | **0.006** |

- Seuil calibré par l'indice de Youden sur la validation (seuil optimal = 0.018)
- Score d'utilité PhysioNet : −0.052 (vs −1.97 avec seuil naïf à 0.5)
- Variance MC Dropout : ×7.4 plus élevée sur les erreurs → incertitude corrélée aux erreurs

### Ablation : ratio de labels

| Ratio | Labels (+) | AUROC | AUPRC | F1 | ECE |
|---|---|---|---|---|---|
| 5% | 108 | 0.8642 | 0.1443 | 0.1410 | 0.0170 |
| **10%** | **215** | **0.8684** | 0.1356 | 0.1228 | **0.0058** |
| 25% | 539 | 0.8669 | **0.1484** | 0.1153 | 0.0179 |
| 50% | 1078 | 0.8679 | 0.1363 | 0.1139 | 0.0088 |

> L'AUROC varie de seulement **0.004 points** sur toute la plage 5–50% : la représentation apprise par diffusion sur les données non-labellées est suffisante, le guidage conditionnel est secondaire.

---

## Dataset

**PhysioNet/CinC Challenge 2019** — disponible sur [Kaggle](https://www.kaggle.com/datasets/salikhussaini49/prediction-of-sepsis)

Placer les données dans `data/raw/sepsis/` :
```
data/raw/sepsis/
├── training_setA/training/*.psv   (20 336 patients)
└── training_setB/training_setB/  (20 000 patients)
```

---

## Installation

```bash
# Cloner le repo
git clone https://github.com/BigYellowData/diffusion-ts-sepsis.git
cd diffusion-ts-sepsis

# Installer les dépendances avec uv
uv sync

# Vérifier le GPU
uv run python -c "import torch; print(torch.cuda.get_device_name(0))"
```

> PyTorch est installé avec support CUDA 12.8. Pour CPU uniquement, retirer les lignes `[[tool.uv.index]]` du `pyproject.toml`.

---

## Utilisation

```bash
# Pipeline complet
uv run python main.py --stage all

# Étapes individuelles
uv run python main.py --stage preprocess   # ~4 min, mis en cache ensuite
uv run python main.py --stage diffusion    # ~25 min (100 epochs, RTX 4060 Ti)
uv run python main.py --stage classifier   # ~15 min avec early stopping
uv run python main.py --stage evaluate     # MC Dropout, 50 passes
uv run python main.py --stage compare      # Comparaison avec les baselines

# Changer le ratio de labels (semi-supervisé)
uv run python main.py --stage all --label_ratio 0.05   # 5% de labels
uv run python main.py --stage all --label_ratio 0.25   # 25% de labels
```

---

## Architecture

```
src/
├── data/
│   ├── preprocess.py     # Chargement parallèle, fenêtrage, normalisation
│   └── dataset.py        # PyTorch Dataset + DataLoaders
├── models/
│   ├── denoiser.py       # Transformer débruiteur (trend/seasonal + MC Dropout)
│   ├── diffusion_ts.py   # DDPM complet (cosine schedule, DDIM, CFG)
│   └── classifier.py     # Transformer + [CLS] token + MC Dropout
├── training/
│   ├── train_diffusion.py   # Entraînement semi-supervisé
│   └── train_classifier.py  # Augmentation + class-balanced loss + early stopping
├── baselines/
│   ├── xgboost_baseline.py
│   ├── lstm_baseline.py
│   └── compare.py
└── utils/
    ├── metrics.py      # AUROC, AUPRC, F1, PhysioNet utility, ECE, abstention curve
    └── uncertainty.py  # Évaluation MC Dropout, corrélation incertitude–erreur
```

---

## Tests

```bash
uv run pytest          # 76 tests, ~1 seconde
uv run pytest -v       # mode verbose
```

---

## Pipeline technique

```
Données brutes (PhysioNet 2019, 40 336 patients)
              ↓
  Prétraitement (fenêtres 24h, masque d'observation, normalisation)
              ↓  125 988 fenêtres  |  2.29% positifs
  Diffusion-TS (semi-supervisé, 100 epochs)
        ↙                        ↘
Génération conditionnelle     Distribution générale
(6 471 synth. sepsis)         des trajectoires
        ↘                        ↙
  Classifieur Transformer + MC Dropout
  (early stop epoch 8, AUROC val = 0.852)
              ↓
  Prédiction + Score d'incertitude
```

---

## Références

- Yuan & Qiao (2024). *Diffusion-TS: Interpretable Diffusion for General Time Series Generation*. ICLR 2024.
- Reyna et al. (2020). *Early Prediction of Sepsis from Clinical Data: The PhysioNet/CinC Challenge 2019*. Critical Care Medicine.
- Gal & Ghahramani (2016). *Dropout as a Bayesian Approximation*. ICML 2016.
- Nichol & Dhariwal (2021). *Improved Denoising Diffusion Probabilistic Models*. ICML 2021.
