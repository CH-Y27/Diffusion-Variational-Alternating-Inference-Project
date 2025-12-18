# models/diffusion_model.py
# A small MLP epsilon-predictor for tabular/high-dim vectors.
from __future__ import annotations

import torch
import torch.nn as nn

from .diffusion_utils import timestep_embedding


class MLPDenoiser(nn.Module):
    """
    Predicts epsilon (noise) given x_t and timestep t.

    Designed to be lightweight and stable on CPU.
    """
    def __init__(self, x_dim: int, hidden: int = 256, n_layers: int = 4, t_embed_dim: int = 128, dropout: float = 0.0):
        super().__init__()
        self.x_dim = x_dim
        self.t_embed_dim = t_embed_dim

        self.t_proj = nn.Sequential(
            nn.Linear(t_embed_dim, hidden),
            nn.SiLU(),
        )

        layers = []
        in_dim = x_dim + hidden
        for _ in range(n_layers - 1):
            layers += [
                nn.Linear(in_dim, hidden),
                nn.SiLU(),
                nn.Dropout(dropout),
            ]
            in_dim = hidden
        layers += [nn.Linear(in_dim, x_dim)]
        self.net = nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x_t: (B, D)
        t:   (B,) int64
        """
        t_emb = timestep_embedding(t, self.t_embed_dim)
        t_h = self.t_proj(t_emb)
        h = torch.cat([x_t, t_h], dim=1)
        return self.net(h)
