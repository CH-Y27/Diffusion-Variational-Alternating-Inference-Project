# models/diffusion_utils.py
# Simple DDPM utilities (CPU-friendly) + inpainting helper.
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class DiffusionSchedule:
    """Precomputes DDPM schedule terms for t in {0,...,T-1}."""
    T: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    device: Optional[torch.device] = None
    dtype: torch.dtype = torch.float32

    def __post_init__(self):
        self.device = self.device or torch.device("cpu")
        betas = torch.linspace(self.beta_start, self.beta_end, self.T, device=self.device, dtype=self.dtype)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        self.betas = betas                    # (T,)
        self.alphas = alphas                  # (T,)
        self.alpha_bar = alpha_bar            # (T,)
        self.sqrt_alpha_bar = torch.sqrt(alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        self.posterior_variance = betas * (1.0 - alpha_bar.roll(1, 0)) / (1.0 - alpha_bar)
        self.posterior_variance[0] = 0.0

    def to(self, device: torch.device, dtype: Optional[torch.dtype] = None) -> "DiffusionSchedule":
        dtype = dtype or self.dtype
        self.device = device
        self.dtype = dtype
        for name in [
            "betas", "alphas", "alpha_bar", "sqrt_alpha_bar", "sqrt_one_minus_alpha_bar",
            "sqrt_recip_alphas", "posterior_variance"
        ]:
            setattr(self, name, getattr(self, name).to(device=device, dtype=dtype))
        return self


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Standard sinusoidal embedding.
    t: (B,) int64 or float tensor in [0, T-1]
    returns: (B, dim)
    """
    if t.dtype != torch.float32 and t.dtype != torch.float64:
        t = t.float()
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(0, half, device=t.device, dtype=t.dtype) / (half - 1))
    args = t[:, None] * freqs[None, :]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
    return emb


def apply_inpainting_constraint(
    x_t: torch.Tensor,
    x0_obs: torch.Tensor,
    obs_mask: torch.Tensor,
    schedule: DiffusionSchedule,
    t_index: torch.Tensor,
    noise: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Enforce observed entries at timestep t by replacing them with the forward noised version
    of the observed x0.

    obs_mask: 1 for observed, 0 for missing (same shape as x_t).
    t_index: (B,) int64 indices.
    """
    if noise is None:
        noise = torch.randn_like(x_t)

    sa = schedule.sqrt_alpha_bar[t_index].view(-1, 1)
    so = schedule.sqrt_one_minus_alpha_bar[t_index].view(-1, 1)
    x_obs_t = sa * x0_obs + so * noise
    return x_t * (1.0 - obs_mask) + x_obs_t * obs_mask
