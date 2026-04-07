"""
Main entry point for the Sepsis Prediction project.

Usage:
    python main.py                        # full pipeline
    python main.py --stage preprocess     # preprocessing only
    python main.py --stage diffusion      # diffusion pre-training only
    python main.py --stage classifier     # classifier training only
    python main.py --stage evaluate       # evaluation + uncertainty analysis
    python main.py --label_ratio 0.05     # override label ratio
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import torch
import yaml

# ─── Logging setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_device(cfg: dict) -> torch.device:
    if cfg["device"] == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    return device


# ─── Pipeline stages ──────────────────────────────────────────────────────────

def stage_preprocess(cfg: dict) -> dict:
    from src.data.preprocess import run_preprocessing
    logger.info("=== STAGE: Preprocessing ===")
    return run_preprocessing(cfg)


def stage_diffusion(cfg: dict, data: dict, device: torch.device) -> None:
    from src.models.diffusion_ts import DiffusionTS
    from src.data.dataset import build_diffusion_loader
    from src.training.train_diffusion import train_diffusion

    logger.info("=== STAGE: Diffusion pre-training ===")
    model = DiffusionTS(cfg)
    loader = build_diffusion_loader(data["splits"], cfg)
    save_dir = cfg["training"]["checkpoint_dir"]
    train_diffusion(model, loader, cfg, device, save_dir=save_dir)


def stage_classifier(cfg: dict, data: dict, device: torch.device) -> None:
    from src.models.diffusion_ts import DiffusionTS
    from src.models.classifier import SepsisClassifier
    from src.data.dataset import build_dataloaders
    from src.training.train_classifier import train_classifier

    logger.info("=== STAGE: Classifier training ===")
    loaders = build_dataloaders(data["splits"], cfg)
    clf_cfg = cfg["classifier"]
    diff_cfg = cfg["diffusion"]

    classifier = SepsisClassifier(
        n_features=diff_cfg["n_features"],
        seq_len=diff_cfg["seq_len"],
        d_model=clf_cfg["d_model"],
        n_heads=clf_cfg["n_heads"],
        n_layers=clf_cfg["n_layers"],
        d_ff=clf_cfg["d_ff"],
        dropout=clf_cfg["dropout"],
    )

    # Load pre-trained diffusion model for augmentation (if checkpoint exists)
    diffusion_model = None
    ckpt = os.path.join(cfg["training"]["checkpoint_dir"], "diffusion_best.pt")
    if os.path.exists(ckpt):
        logger.info(f"Loading diffusion checkpoint from '{ckpt}'")
        diffusion_model = DiffusionTS(cfg).to(device)
        diffusion_model.load_state_dict(torch.load(ckpt, map_location=device))
    else:
        logger.warning("No diffusion checkpoint found – skipping augmentation")

    train_classifier(
        classifier,
        loaders["train"],
        loaders["val"],
        cfg,
        device,
        diffusion_model=diffusion_model,
        save_dir=cfg["training"]["checkpoint_dir"],
    )


def stage_compare(cfg: dict, data: dict, device: torch.device) -> None:
    from src.baselines.compare import run_comparison
    logger.info("=== STAGE: Baseline comparison ===")
    run_comparison(data["splits"], cfg, device)


def stage_plots() -> None:
    from src.utils.plots import generate_all_plots
    logger.info("=== STAGE: Generating figures ===")
    generate_all_plots(results_dir="results")


def stage_evaluate(cfg: dict, data: dict, device: torch.device) -> None:
    from src.models.classifier import SepsisClassifier
    from src.data.dataset import build_dataloaders
    from src.utils.uncertainty import evaluate_with_uncertainty, uncertainty_correlation, save_results

    logger.info("=== STAGE: Evaluation with MC Dropout ===")
    loaders = build_dataloaders(data["splits"], cfg)
    clf_cfg = cfg["classifier"]
    diff_cfg = cfg["diffusion"]

    classifier = SepsisClassifier(
        n_features=diff_cfg["n_features"],
        seq_len=diff_cfg["seq_len"],
        d_model=clf_cfg["d_model"],
        n_heads=clf_cfg["n_heads"],
        n_layers=clf_cfg["n_layers"],
        d_ff=clf_cfg["d_ff"],
        dropout=clf_cfg["dropout"],
    )

    ckpt = os.path.join(cfg["training"]["checkpoint_dir"], "classifier_best.pt")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Classifier checkpoint not found at '{ckpt}'")
    classifier.load_state_dict(torch.load(ckpt, map_location=device))
    classifier.to(device)

    # MC Dropout evaluation on test set (threshold calibrated on val set)
    results = evaluate_with_uncertainty(
        classifier,
        loaders["test"],
        device,
        n_mc_samples=clf_cfg["mc_samples"],
        val_loader=loaders["val"],
    )

    # Uncertainty–error correlation
    unc_stats = uncertainty_correlation(
        results["y_true"], results["mean_prob"], results["uncertainty"]
    )
    results["metrics"].update(unc_stats)

    save_results(results, out_dir="results")

    # Print summary table
    m = results["metrics"]
    print("\n" + "=" * 50)
    print("TEST SET RESULTS")
    print("=" * 50)
    print(f"  Threshold      : {m.get('threshold', 0.5):.4f}  (Youden, calibrated on val)")
    print(f"  AUROC          : {m['auroc']:.4f}")
    print(f"  AUPRC          : {m['auprc']:.4f}")
    print(f"  F1             : {m['f1']:.4f}")
    print(f"  Brier          : {m['brier']:.4f}")
    print(f"  PhysioNet Util : {m['physionet_utility']:.4f}")
    print(f"  ECE            : {m['ece']:.4f}")
    if "auac" in m:
        print(f"  AUAC           : {m['auac']:.4f}")
    print(f"  Unc–Err corr   : {m['corr_uncertainty_error']:.3f}")
    print("=" * 50)


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Sepsis Prediction Pipeline")
    parser.add_argument(
        "--stage",
        choices=["preprocess", "diffusion", "classifier", "evaluate", "compare", "plots", "all"],
        default="all",
        help="Pipeline stage to run",
    )
    parser.add_argument("--config", default="configs/config.yaml", help="Config file path")
    parser.add_argument(
        "--label_ratio",
        type=float,
        default=None,
        help="Override label_ratio in config (e.g. 0.05, 0.10, 0.25, 0.50)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    if args.label_ratio is not None:
        cfg["data"]["label_ratio"] = args.label_ratio
        logger.info(f"Label ratio overridden to {args.label_ratio}")

    device = get_device(cfg)

    # Make checkpoint / log dirs
    Path(cfg["training"]["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["training"]["log_dir"]).mkdir(parents=True, exist_ok=True)

    data = None
    if args.stage in ("preprocess", "all"):
        data = stage_preprocess(cfg)

    if args.stage in ("diffusion", "classifier", "evaluate", "compare", "all"):
        if data is None:
            data = stage_preprocess(cfg)   # loads from cache if available

    if args.stage in ("diffusion", "all"):
        stage_diffusion(cfg, data, device)

    if args.stage in ("classifier", "all"):
        stage_classifier(cfg, data, device)

    if args.stage in ("evaluate", "all"):
        stage_evaluate(cfg, data, device)

    if args.stage == "compare":
        stage_compare(cfg, data, device)

    if args.stage in ("plots", "all"):
        stage_plots()


if __name__ == "__main__":
    main()
