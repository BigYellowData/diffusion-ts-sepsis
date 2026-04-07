"""
Diffusion-TS: DDPM-based generative model for clinical time series.

Implements:
  - Cosine / linear noise schedule
  - Forward diffusion  q(x_t | x_0)
  - Reverse denoising  p_θ(x_{t-1} | x_t)
  - Classifier-free guidance for conditional generation
  - Sampling routines (DDPM + fast DDIM)
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .denoiser import DiffusionTSDenoiser


# ─── Noise schedules ──────────────────────────────────────────────────────────

def linear_beta_schedule(T: int, beta_start: float, beta_end: float) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, T)


def cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    """Cosine schedule as proposed by Nichol & Dhariwal (2021)."""
    steps = T + 1
    x = torch.linspace(0, T, steps)
    alphas_cumprod = torch.cos(((x / T) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0001, 0.9999)


# ─── Main model ───────────────────────────────────────────────────────────────

class DiffusionTS(nn.Module):
    """
    Full DDPM model wrapping the Transformer denoiser.

    Key methods:
        loss(x0, cond)           – training loss (conditional or unconditional)
        sample(shape, cond)      – DDPM ancestral sampling
        sample_ddim(shape, cond) – faster DDIM sampling
    """

    UNCOND_TOKEN = 2   # class index reserved for unconditional

    def __init__(self, cfg: dict):
        super().__init__()
        diff_cfg = cfg["diffusion"]

        self.T = diff_cfg["n_diffusion_steps"]
        self.n_features = diff_cfg["n_features"]
        self.seq_len = diff_cfg["seq_len"]
        self.guidance_scale = cfg["generation"]["guidance_scale"]

        # Denoiser
        self.denoiser = DiffusionTSDenoiser(
            n_features=diff_cfg["n_features"],
            seq_len=diff_cfg["seq_len"],
            d_model=diff_cfg["d_model"],
            n_heads=diff_cfg["n_heads"],
            n_layers=diff_cfg["n_layers"],
            d_ff=diff_cfg["d_ff"],
            dropout=diff_cfg["dropout"],
        )

        # Register noise schedule buffers
        if diff_cfg["scheduler"] == "cosine":
            betas = cosine_beta_schedule(self.T)
        else:
            betas = linear_beta_schedule(
                self.T, diff_cfg["beta_start"], diff_cfg["beta_end"]
            )

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", alphas_cumprod.sqrt())
        self.register_buffer("sqrt_one_minus_alphas_cumprod", (1 - alphas_cumprod).sqrt())
        self.register_buffer(
            "posterior_variance",
            betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod),
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _extract(self, a: torch.Tensor, t: torch.Tensor, shape) -> torch.Tensor:
        """Gather schedule values at time steps t and broadcast to shape."""
        out = a.gather(0, t)
        return out.reshape(t.shape[0], *((1,) * (len(shape) - 1)))

    # ── Forward process ───────────────────────────────────────────────────────

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample x_t ~ q(x_t | x_0) using the reparameterisation trick.
        Returns (x_t, noise).
        """
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_one_minus = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        x_t = sqrt_alpha * x0 + sqrt_one_minus * noise
        return x_t, noise

    # ── Training loss ─────────────────────────────────────────────────────────

    def loss(
        self,
        x0: torch.Tensor,
        cond: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        p_uncond: float = 0.1,
    ) -> torch.Tensor:
        """
        Simple ε-prediction loss.

        Args:
            x0      : (B, T, F) clean time series
            cond    : (B,) class labels (None → unconditional training)
            mask    : (B, T, F) observation mask (1=observed). Loss weighted by mask.
            p_uncond: probability of dropping the condition (classifier-free guidance)
        """
        B = x0.shape[0]
        t = torch.randint(0, self.T, (B,), device=x0.device)
        x_t, noise = self.q_sample(x0, t)

        # Classifier-free guidance: randomly null-out the condition
        if cond is not None and p_uncond > 0:
            drop = torch.rand(B, device=x0.device) < p_uncond
            cond = cond.clone()
            cond[drop] = self.UNCOND_TOKEN

        pred_noise = self.denoiser(x_t, t, cond)

        if mask is not None:
            # Weight loss by observation mask (observed timesteps count more)
            weight = mask.float() * 0.5 + 0.5   # still count imputed but half weight
            loss = (F.mse_loss(pred_noise, noise, reduction="none") * weight).mean()
        else:
            loss = F.mse_loss(pred_noise, noise)
        return loss

    # ── Reverse step ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def p_sample(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """One DDPM reverse step: x_{t-1} ~ p_θ(x_{t-1} | x_t)."""
        # Classifier-free guidance: combine conditional + unconditional predictions
        eps_cond = self.denoiser(x_t, t, cond)
        if cond is not None and self.guidance_scale != 1.0:
            uncond = torch.full_like(cond, self.UNCOND_TOKEN)
            eps_uncond = self.denoiser(x_t, t, uncond)
            eps = eps_uncond + self.guidance_scale * (eps_cond - eps_uncond)
        else:
            eps = eps_cond

        # Compute mean
        betas_t = self._extract(self.betas, t, x_t.shape)
        sqrt_recip = self._extract((1.0 / self.alphas).sqrt(), t, x_t.shape)
        sqrt_one_minus = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        model_mean = sqrt_recip * (x_t - betas_t / sqrt_one_minus * eps)

        # Add noise for t > 0
        noise = torch.randn_like(x_t)
        posterior_var = self._extract(self.posterior_variance, t, x_t.shape)
        nonzero = (t > 0).float().reshape(-1, 1, 1)
        return model_mean + nonzero * posterior_var.sqrt() * noise

    # ── Full DDPM sampling ────────────────────────────────────────────────────

    @torch.no_grad()
    def sample(
        self,
        n_samples: int,
        cond: torch.Tensor | None = None,
        device: str | torch.device = "cpu",
    ) -> torch.Tensor:
        """
        Generate n_samples via full DDPM ancestral sampling.
        cond: (n_samples,) class labels or None.
        Returns (n_samples, T, F).
        """
        shape = (n_samples, self.seq_len, self.n_features)
        x = torch.randn(shape, device=device)

        for step in reversed(range(self.T)):
            t = torch.full((n_samples,), step, device=device, dtype=torch.long)
            x = self.p_sample(x, t, cond)
        return x

    # ── Fast DDIM sampling ────────────────────────────────────────────────────

    @torch.no_grad()
    def sample_ddim(
        self,
        n_samples: int,
        cond: torch.Tensor | None = None,
        n_steps: int = 50,
        eta: float = 0.0,
        device: str | torch.device = "cpu",
    ) -> torch.Tensor:
        """
        DDIM sampling (Song et al., 2020) – much faster than full DDPM.
        eta=0 → deterministic; eta=1 → equivalent to DDPM.
        """
        # Subsample timesteps uniformly
        step_size = self.T // n_steps
        timesteps = list(reversed(range(0, self.T, step_size)))[:n_steps]

        x = torch.randn(n_samples, self.seq_len, self.n_features, device=device)

        for i, step in enumerate(timesteps):
            t = torch.full((n_samples,), step, device=device, dtype=torch.long)
            prev_step = timesteps[i + 1] if i + 1 < len(timesteps) else 0

            alpha_t = self.alphas_cumprod[step]
            alpha_prev = self.alphas_cumprod[prev_step]

            eps_cond = self.denoiser(x, t, cond)
            if cond is not None and self.guidance_scale != 1.0:
                uncond = torch.full_like(cond, self.UNCOND_TOKEN)
                eps_uncond = self.denoiser(x, t, uncond)
                eps = eps_uncond + self.guidance_scale * (eps_cond - eps_uncond)
            else:
                eps = eps_cond

            # Predicted x0
            x0_pred = (x - (1 - alpha_t).sqrt() * eps) / alpha_t.sqrt()
            x0_pred = x0_pred.clamp(-3, 3)

            # Direction
            sigma = eta * ((1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev)).sqrt()
            dir_xt = (1 - alpha_prev - sigma ** 2).sqrt() * eps

            noise = sigma * torch.randn_like(x)
            x = alpha_prev.sqrt() * x0_pred + dir_xt + noise

        return x

    # ── Conditional generation for data augmentation ──────────────────────────

    @torch.no_grad()
    def generate_class(
        self,
        label: int,
        n_samples: int,
        device: str | torch.device = "cpu",
        fast: bool = True,
        n_ddim_steps: int = 50,
    ) -> torch.Tensor:
        """Generate n_samples conditioned on a given class label."""
        cond = torch.full((n_samples,), label, dtype=torch.long, device=device)
        if fast:
            return self.sample_ddim(n_samples, cond, n_steps=n_ddim_steps, device=device)
        return self.sample(n_samples, cond, device=device)
