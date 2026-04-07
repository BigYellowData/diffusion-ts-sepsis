"""Tests for src/models/ (denoiser, diffusion_ts, classifier)."""

import pytest
import torch
import torch.nn as nn

from src.models.denoiser import (
    SinusoidalEmbedding,
    SeriesDecompositionBlock,
    DiffusionTSDenoiser,
    enable_mc_dropout,
)
from src.models.diffusion_ts import DiffusionTS
from src.models.classifier import SepsisClassifier, mc_predict


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_batch(cfg, B=4):
    F = cfg["diffusion"]["n_features"]
    T = cfg["diffusion"]["seq_len"]
    x = torch.randn(B, T, F)
    mask = torch.ones(B, T, F)
    return x, mask


# ── SinusoidalEmbedding ───────────────────────────────────────────────────────

class TestSinusoidalEmbedding:
    def test_output_shape(self, cfg):
        d = cfg["diffusion"]["d_model"]
        emb = SinusoidalEmbedding(d)
        t = torch.arange(4)
        out = emb(t)
        assert out.shape == (4, d)

    def test_different_steps_differ(self, cfg):
        emb = SinusoidalEmbedding(cfg["diffusion"]["d_model"])
        t0 = emb(torch.tensor([0]))
        t1 = emb(torch.tensor([1]))
        assert not torch.allclose(t0, t1)


# ── SeriesDecompositionBlock ──────────────────────────────────────────────────

class TestSeriesDecomposition:
    def test_output_shapes(self, cfg):
        B, T, F = 4, cfg["diffusion"]["seq_len"], cfg["diffusion"]["n_features"]
        block = SeriesDecompositionBlock(kernel_size=5)
        x = torch.randn(B, T, F)
        seasonal, trend = block(x)
        assert seasonal.shape == x.shape
        assert trend.shape == x.shape

    def test_seasonal_plus_trend_equals_input(self, cfg):
        block = SeriesDecompositionBlock(kernel_size=5)
        x = torch.randn(4, cfg["diffusion"]["seq_len"], cfg["diffusion"]["n_features"])
        seasonal, trend = block(x)
        torch.testing.assert_close(seasonal + trend, x)


# ── DiffusionTSDenoiser ───────────────────────────────────────────────────────

class TestDenoiser:
    def test_output_shape(self, cfg):
        dcfg = cfg["diffusion"]
        denoiser = DiffusionTSDenoiser(
            n_features=dcfg["n_features"],
            seq_len=dcfg["seq_len"],
            d_model=dcfg["d_model"],
            n_heads=dcfg["n_heads"],
            n_layers=dcfg["n_layers"],
            d_ff=dcfg["d_ff"],
            dropout=dcfg["dropout"],
        )
        B = 4
        x_t = torch.randn(B, dcfg["seq_len"], dcfg["n_features"])
        t = torch.randint(0, 10, (B,))
        out = denoiser(x_t, t)
        assert out.shape == x_t.shape

    def test_conditional_output_shape(self, cfg):
        dcfg = cfg["diffusion"]
        denoiser = DiffusionTSDenoiser(
            n_features=dcfg["n_features"],
            seq_len=dcfg["seq_len"],
            d_model=dcfg["d_model"],
            n_heads=dcfg["n_heads"],
            n_layers=dcfg["n_layers"],
            d_ff=dcfg["d_ff"],
            dropout=dcfg["dropout"],
        )
        B = 4
        x_t = torch.randn(B, dcfg["seq_len"], dcfg["n_features"])
        t = torch.randint(0, 10, (B,))
        cond = torch.randint(0, 2, (B,))
        out = denoiser(x_t, t, cond)
        assert out.shape == x_t.shape

    def test_no_nan_in_output(self, cfg):
        dcfg = cfg["diffusion"]
        denoiser = DiffusionTSDenoiser(
            n_features=dcfg["n_features"], seq_len=dcfg["seq_len"],
            d_model=dcfg["d_model"], n_heads=dcfg["n_heads"],
            n_layers=dcfg["n_layers"], d_ff=dcfg["d_ff"], dropout=0.0,
        )
        x_t = torch.randn(2, dcfg["seq_len"], dcfg["n_features"])
        t = torch.zeros(2, dtype=torch.long)
        out = denoiser(x_t, t)
        assert not torch.isnan(out).any()


# ── DiffusionTS ───────────────────────────────────────────────────────────────

class TestDiffusionTS:
    def test_loss_is_scalar(self, cfg):
        model = DiffusionTS(cfg)
        x, _ = make_batch(cfg)
        loss = model.loss(x)
        assert loss.ndim == 0
        assert loss.item() > 0

    def test_loss_with_condition(self, cfg):
        model = DiffusionTS(cfg)
        x, _ = make_batch(cfg, B=4)
        cond = torch.randint(0, 2, (4,))
        loss = model.loss(x, cond=cond)
        assert loss.item() > 0

    def test_loss_with_mask(self, cfg):
        model = DiffusionTS(cfg)
        x, mask = make_batch(cfg)
        loss = model.loss(x, mask=mask)
        assert not torch.isnan(loss)

    def test_q_sample_shape(self, cfg):
        model = DiffusionTS(cfg)
        x, _ = make_batch(cfg)
        t = torch.zeros(4, dtype=torch.long)
        x_t, noise = model.q_sample(x, t)
        assert x_t.shape == x.shape
        assert noise.shape == x.shape

    def test_q_sample_at_t0_close_to_x0(self, cfg):
        """At t=0, x_t = sqrt_alpha[0]*x0 + sqrt_one_minus[0]*noise.
        We verify the signal coefficient dominates the noise coefficient."""
        model = DiffusionTS(cfg)
        alpha0 = model.sqrt_alphas_cumprod[0].item()
        sigma0 = model.sqrt_one_minus_alphas_cumprod[0].item()
        assert alpha0 > sigma0, "signal should dominate noise at t=0"

    def test_ddim_sample_shape(self, cfg):
        model = DiffusionTS(cfg)
        samples = model.sample_ddim(n_samples=2, n_steps=3, device=torch.device("cpu"))
        assert samples.shape == (2, cfg["diffusion"]["seq_len"], cfg["diffusion"]["n_features"])

    def test_generate_class_shape(self, cfg):
        model = DiffusionTS(cfg)
        samples = model.generate_class(label=1, n_samples=3, fast=True, n_ddim_steps=3)
        assert samples.shape == (3, cfg["diffusion"]["seq_len"], cfg["diffusion"]["n_features"])

    def test_loss_backward(self, cfg):
        model = DiffusionTS(cfg)
        x, _ = make_batch(cfg)
        loss = model.loss(x)
        loss.backward()   # must not raise


# ── SepsisClassifier ──────────────────────────────────────────────────────────

class TestSepsisClassifier:
    def _make_clf(self, cfg):
        ccfg = cfg["classifier"]
        dcfg = cfg["diffusion"]
        return SepsisClassifier(
            n_features=dcfg["n_features"],
            seq_len=dcfg["seq_len"],
            d_model=ccfg["d_model"],
            n_heads=ccfg["n_heads"],
            n_layers=ccfg["n_layers"],
            d_ff=ccfg["d_ff"],
            dropout=ccfg["dropout"],
        )

    def test_output_shape(self, cfg):
        clf = self._make_clf(cfg)
        x, mask = make_batch(cfg, B=8)
        logit = clf(x, mask)
        assert logit.shape == (8,)

    def test_output_without_mask(self, cfg):
        clf = self._make_clf(cfg)
        x, _ = make_batch(cfg, B=4)
        logit = clf(x, mask=None)
        assert logit.shape == (4,)

    def test_no_nan_in_output(self, cfg):
        clf = self._make_clf(cfg)
        x, mask = make_batch(cfg)
        logit = clf(x, mask)
        assert not torch.isnan(logit).any()

    def test_backward(self, cfg):
        clf = self._make_clf(cfg)
        x, mask = make_batch(cfg)
        logit = clf(x, mask)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logit, torch.zeros(logit.shape[0])
        )
        loss.backward()

    def test_mc_predict_shapes(self, cfg):
        clf = self._make_clf(cfg)
        x, mask = make_batch(cfg, B=6)
        n_mc = cfg["classifier"]["mc_samples"]
        out = mc_predict(clf, x, mask, n_samples=n_mc)
        assert out["mean_prob"].shape == (6,)
        assert out["uncertainty"].shape == (6,)
        assert out["logits"].shape == (6, n_mc)

    def test_mc_uncertainty_nonnegative(self, cfg):
        clf = self._make_clf(cfg)
        x, mask = make_batch(cfg)
        out = mc_predict(clf, x, mask, n_samples=cfg["classifier"]["mc_samples"])
        assert (out["uncertainty"] >= 0).all()

    def test_mc_mean_prob_in_01(self, cfg):
        clf = self._make_clf(cfg)
        x, mask = make_batch(cfg)
        out = mc_predict(clf, x, mask, n_samples=cfg["classifier"]["mc_samples"])
        assert (out["mean_prob"] >= 0).all()
        assert (out["mean_prob"] <= 1).all()

    def test_enable_mc_dropout_keeps_dropout_in_train_mode(self, cfg):
        from src.models.denoiser import enable_mc_dropout
        clf = self._make_clf(cfg)
        enable_mc_dropout(clf)
        for m in clf.modules():
            if isinstance(m, nn.Dropout):
                assert m.training
