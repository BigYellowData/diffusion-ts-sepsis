"""
Transformer-based denoiser for Diffusion-TS.

Architecture (simplified Diffusion-TS):
  - Sinusoidal time-step embedding
  - Trend + Seasonality decomposition blocks (as in the paper)
  - Multi-head self-attention with MC Dropout
  - Output projection back to (T, F)

MC Dropout: dropout is applied inside every Transformer block.
At inference we call model.train() selectively on dropout layers so
stochastic passes yield an uncertainty estimate.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ─── Sinusoidal diffusion-step embedding ─────────────────────────────────────

class SinusoidalEmbedding(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t : (B,) integer diffusion steps
        returns (B, d_model)
        """
        half = self.d_model // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)   # (B, half)
        emb = torch.cat([args.sin(), args.cos()], dim=-1)     # (B, d_model)
        return emb


# ─── Trend decomposition block ────────────────────────────────────────────────

class MovingAverage(nn.Module):
    """Applies a causal moving average to extract the trend component."""

    def __init__(self, kernel_size: int = 5):
        super().__init__()
        self.kernel_size = kernel_size
        # Padding on the left to keep causality
        self.pad = nn.ConstantPad1d((kernel_size - 1, 0), 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, F)  → trend (B, T, F)"""
        x_t = rearrange(x, "b t f -> b f t")
        x_padded = self.pad(x_t)
        trend = F.avg_pool1d(x_padded, kernel_size=self.kernel_size, stride=1)
        return rearrange(trend, "b f t -> b t f")


class SeriesDecompositionBlock(nn.Module):
    def __init__(self, kernel_size: int = 5):
        super().__init__()
        self.moving_avg = MovingAverage(kernel_size)

    def forward(self, x: torch.Tensor):
        trend = self.moving_avg(x)
        seasonal = x - trend
        return seasonal, trend


# ─── Transformer encoder block with MC Dropout ───────────────────────────────

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask=None) -> torch.Tensor:
        # Self-attention + residual
        attn_out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
        x = self.norm1(x + self.drop(attn_out))
        # Feed-forward + residual
        x = self.norm2(x + self.ff(x))
        return x


# ─── Denoiser ─────────────────────────────────────────────────────────────────

class DiffusionTSDenoiser(nn.Module):
    """
    Denoiser ε_θ(x_t, t, c) for Diffusion-TS.

    Inputs:
        x_t  : (B, T, F) noisy time series
        t    : (B,)       diffusion step indices
        cond : (B, d_cond) optional class conditioning (None → unconditional)
    Output:
        (B, T, F) predicted noise
    """

    def __init__(
        self,
        n_features: int,
        seq_len: int,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: int = 256,
        dropout: float = 0.1,
        n_classes: int = 2,             # for classifier-free guidance
        trend_kernel: int = 5,
    ):
        super().__init__()
        self.n_features = n_features
        self.seq_len = seq_len
        self.d_model = d_model

        # Input projection
        self.input_proj = nn.Linear(n_features, d_model)

        # Positional encoding (learnable)
        self.pos_emb = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

        # Diffusion-step embedding
        self.time_emb = SinusoidalEmbedding(d_model)
        self.time_proj = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model),
        )

        # Class conditioning embedding (label 0=non-sepsis, 1=sepsis, 2=unconditional)
        self.class_emb = nn.Embedding(n_classes + 1, d_model)

        # Series decomposition
        self.decomp = SeriesDecompositionBlock(kernel_size=trend_kernel)

        # Transformer backbone
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Trend pathway (lightweight MLP per feature)
        self.trend_proj = nn.Linear(d_model, d_model)
        self.trend_out = nn.Linear(d_model, n_features)

        # Seasonal pathway → final output projection
        self.out_proj = nn.Linear(d_model, n_features)

    # ── Conditioning helpers ──────────────────────────────────────────────────

    def _encode_condition(self, t: torch.Tensor, cond: torch.Tensor | None) -> torch.Tensor:
        """Merge time embedding + optional class embedding → (B, 1, d_model)."""
        t_emb = self.time_proj(self.time_emb(t))          # (B, d_model)
        if cond is not None:
            c_emb = self.class_emb(cond.long())            # (B, d_model)
            t_emb = t_emb + c_emb
        return t_emb.unsqueeze(1)                           # (B, 1, d_model)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T, F = x_t.shape

        # Decompose noisy input
        seasonal, trend_in = self.decomp(x_t)

        # Project to model dimension
        h = self.input_proj(seasonal) + self.pos_emb[:, :T, :]

        # Add conditioning as an extra token at position 0
        cond_token = self._encode_condition(t, cond)       # (B, 1, d_model)
        h = torch.cat([cond_token, h], dim=1)              # (B, T+1, d_model)

        for block in self.blocks:
            h = block(h)

        # Remove conditioning token
        h = h[:, 1:, :]                                    # (B, T, d_model)

        # Seasonal output
        seasonal_out = self.out_proj(h)                    # (B, T, F)

        # Trend output (separate light branch)
        trend_h = self.trend_proj(h)
        trend_out = self.trend_out(trend_h)                # (B, T, F)

        return seasonal_out + trend_out                    # predicted noise ε


# ─── MC Dropout inference helper ─────────────────────────────────────────────

def enable_mc_dropout(model: nn.Module) -> None:
    """Switch dropout layers to train mode while keeping the rest in eval mode.
    Call this before running multiple stochastic forward passes."""
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()
