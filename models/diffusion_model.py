# models/diffusion_model.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from .diffusion_utils import EDMLoss


class PositionalEmbedding(nn.Module):
    def __init__(self, num_channels: int, max_positions: int = 10000, endpoint: bool = False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        half = self.num_channels // 2
        freqs = torch.arange(0, half, device=x.device, dtype=torch.float32)
        denom = (half - (1 if self.endpoint else 0))
        freqs = freqs / max(1, denom)
        freqs = (1.0 / self.max_positions) ** freqs
        x = x.to(torch.float32).unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([x.cos(), x.sin()], dim=1)
        if self.num_channels % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class MLPDiffusion(nn.Module):
    def __init__(self, d_in: int, dim_t: int = 512):
        super().__init__()
        self.map_noise = PositionalEmbedding(num_channels=dim_t)
        self.time_embed = nn.Sequential(nn.Linear(dim_t, dim_t), nn.SiLU(), nn.Linear(dim_t, dim_t))
        self.proj = nn.Linear(d_in, dim_t)
        self.mlp = nn.Sequential(
            nn.Linear(dim_t, dim_t * 2), nn.SiLU(),
            nn.Linear(dim_t * 2, dim_t * 2), nn.SiLU(),
            nn.Linear(dim_t * 2, dim_t), nn.SiLU(),
            nn.Linear(dim_t, d_in),
        )

    def forward(self, x: torch.Tensor, noise_labels: torch.Tensor) -> torch.Tensor:
        emb = self.map_noise(noise_labels)
        emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape_as(emb)
        emb = self.time_embed(emb)
        h = self.proj(x) + emb
        return self.mlp(h)


class Precond(nn.Module):
    def __init__(self, denoise_fn: nn.Module, sigma_data: float = 0.5, sigma_min: float = 0.0, sigma_max: float = float("inf")):
        super().__init__()
        self.denoise_fn = denoise_fn
        self.sigma_data = float(sigma_data)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1)

        c_skip = (self.sigma_data ** 2) / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / torch.sqrt(sigma ** 2 + self.sigma_data ** 2)
        c_in = 1.0 / torch.sqrt(self.sigma_data ** 2 + sigma ** 2)
        c_noise = sigma.log() / 4.0

        x_in = c_in * x
        F_x = self.denoise_fn(x_in, c_noise.flatten())
        return c_skip * x + c_out * F_x

    def round_sigma(self, sigma: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(sigma)


class EDMModel(nn.Module):
    def __init__(self, d_in: int, hid_dim: int = 512, P_mean: float = -1.2, P_std: float = 1.2, sigma_data: float = 0.5):
        super().__init__()
        denoise_fn = MLPDiffusion(d_in=d_in, dim_t=hid_dim)
        self.net = Precond(denoise_fn, sigma_data=sigma_data)
        self.loss_fn = EDMLoss(P_mean=P_mean, P_std=P_std, sigma_data=sigma_data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(self.net, x)
