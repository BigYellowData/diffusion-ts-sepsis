"""Tests for src/data/dataset.py"""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from src.data.dataset import SepsisDataset, build_dataloaders, build_diffusion_loader


class TestSepsisDataset:
    def test_len(self, synthetic_splits):
        ds = SepsisDataset(**{k: synthetic_splits["train"][k] for k in ("X", "M", "y")})
        assert len(ds) == len(synthetic_splits["train"]["X"])

    def test_getitem_keys(self, synthetic_splits):
        ds = SepsisDataset(**{k: synthetic_splits["train"][k] for k in ("X", "M", "y")})
        item = ds[0]
        assert set(item.keys()) == {"x", "mask", "y", "labelled"}

    def test_tensor_types(self, synthetic_splits):
        ds = SepsisDataset(**{k: synthetic_splits["train"][k] for k in ("X", "M", "y")})
        item = ds[0]
        assert isinstance(item["x"], torch.Tensor)
        assert isinstance(item["y"], torch.Tensor)

    def test_labelled_mask_applied(self, synthetic_splits):
        split = synthetic_splits["train"]
        mask = split["labelled_mask"]
        ds = SepsisDataset(split["X"], split["M"], split["y"], labelled_mask=mask)
        # Gather all labelled values
        labelled_vals = torch.stack([ds[i]["labelled"] for i in range(len(ds))])
        expected = torch.from_numpy(mask.astype(np.float32))
        assert torch.equal(labelled_vals, expected)

    def test_default_all_labelled(self, synthetic_splits):
        split = synthetic_splits["val"]
        ds = SepsisDataset(split["X"], split["M"], split["y"])
        labelled_vals = torch.stack([ds[i]["labelled"] for i in range(len(ds))])
        assert labelled_vals.all()

    def test_shapes(self, synthetic_splits, cfg):
        split = synthetic_splits["train"]
        ds = SepsisDataset(split["X"], split["M"], split["y"])
        item = ds[0]
        assert item["x"].shape == (cfg["data"]["window_size"], cfg["diffusion"]["n_features"])
        assert item["mask"].shape == item["x"].shape


class TestDataLoaders:
    def test_build_dataloaders_keys(self, synthetic_splits, cfg):
        loaders = build_dataloaders(synthetic_splits, cfg)
        assert set(loaders.keys()) == {"train", "val", "test"}

    def test_batch_shapes(self, synthetic_splits, cfg):
        loaders = build_dataloaders(synthetic_splits, cfg)
        batch = next(iter(loaders["val"]))
        B = batch["x"].shape[0]
        assert batch["x"].shape == (B, cfg["data"]["window_size"], cfg["diffusion"]["n_features"])
        assert batch["y"].shape == (B,)

    def test_diffusion_loader(self, synthetic_splits, cfg):
        loader = build_diffusion_loader(synthetic_splits, cfg)
        assert isinstance(loader, DataLoader)
        batch = next(iter(loader))
        assert "x" in batch
