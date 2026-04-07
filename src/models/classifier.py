"""
Transformer classifier with Monte Carlo Dropout for sepsis prediction.

At training time: standard Dropout for regularisation.
At inference time: call enable_mc_dropout() then run N forward passes
to obtain a distribution of predictions → variance = uncertainty.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .denoiser import TransformerBlock, enable_mc_dropout


class SepsisClassifier(nn.Module):
    """
    Inputs:
        x    : (B, T, F)  normalised feature windows
        mask : (B, T, F)  observation mask (optional, concatenated to features)

    Output:
        logit : (B,)  raw logit for sepsis (positive class)
    """

    def __init__(
        self,
        n_features: int,
        seq_len: int,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.2,
        use_mask_features: bool = True,
    ):
        super().__init__()
        self.use_mask_features = use_mask_features
        in_dim = n_features * 2 if use_mask_features else n_features

        # Input projection
        self.input_proj = nn.Linear(in_dim, d_model)

        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Positional encoding (learnable)
        self.pos_emb = nn.Parameter(torch.randn(1, seq_len + 1, d_model) * 0.02)

        # Transformer backbone (with Dropout inside each block)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

        # Classification head
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Returns raw logit (B,)."""
        if self.use_mask_features:
            if mask is None:
                mask = torch.ones_like(x)
            x = torch.cat([x, mask], dim=-1)           # (B, T, 2F)

        h = self.input_proj(x)                          # (B, T, d_model)

        # Prepend [CLS] token
        cls = self.cls_token.expand(h.shape[0], -1, -1) # (B, 1, d_model)
        h = torch.cat([cls, h], dim=1)                  # (B, T+1, d_model)
        h = h + self.pos_emb[:, :h.shape[1], :]

        for block in self.blocks:
            h = block(h)

        cls_out = self.norm(h[:, 0, :])                 # (B, d_model) – CLS
        cls_out = self.drop(cls_out)
        logit = self.head(cls_out).squeeze(-1)          # (B,)
        return logit


# ─── MC Dropout inference ─────────────────────────────────────────────────────

@torch.no_grad()
def mc_predict(
    model: SepsisClassifier,
    x: torch.Tensor,
    mask: torch.Tensor | None,
    n_samples: int = 50,
) -> dict:
    """
    Run n_samples stochastic forward passes with MC Dropout active.

    Returns:
        {
          "mean_prob"   : (B,)   mean predicted probability
          "uncertainty" : (B,)   predictive variance (epistemic uncertainty)
          "logits"      : (B, n_samples)  all raw logits
        }
    """
    enable_mc_dropout(model)

    logits_list = []
    for _ in range(n_samples):
        logit = model(x, mask)                          # (B,)
        logits_list.append(logit)

    logits = torch.stack(logits_list, dim=1)            # (B, n_samples)
    probs = torch.sigmoid(logits)                       # (B, n_samples)

    mean_prob = probs.mean(dim=1)                       # (B,)
    uncertainty = probs.var(dim=1)                      # (B,)  predictive variance

    return {
        "mean_prob": mean_prob,
        "uncertainty": uncertainty,
        "logits": logits,
    }
