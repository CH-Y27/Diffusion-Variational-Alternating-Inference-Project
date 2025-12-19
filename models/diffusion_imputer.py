# models/diffusion_imputer.py
# EDM-based diffusion imputer wrapper (fits your existing diffusion_model.py + diffusion_utils.py)
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
import copy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .diffusion_model import EDMModel
from .diffusion_utils import impute_mask


@dataclass
class DiffusionImputerConfig:
    # keep these fields for compatibility with main.py/configs.py (some may be unused in EDM)
    T: int = 200
    beta_start: float = 1e-4
    beta_end: float = 2e-2

    hidden: int = 64           # will be used as EDM hid_dim
    n_layers: int = 3          # unused by current EDM MLPDiffusion, kept for compatibility
    t_embed_dim: int = 64      # unused by current EDM, kept for compatibility
    dropout: float = 0.0       # unused by current EDM, kept for compatibility

    lr: float = 2e-4
    batch_size: int = 256
    epochs: int = 120
    grad_clip: float = 1.0
    ema: float = 0.999         # EMA decay for stable sampling

    sample_steps: int = 30     # used as EDM sampling num_steps
    num_impute_samples: int = 8
    clamp_val: float = 6.0

    inner_gibbs: int = 1       # used by impute_mask()


class DiffusionImputer:
    """
    Wrapper around EDMModel + impute_mask() for inpainting imputation.

    - fit(): trains EDM loss on completed standardized X
    - impute(): performs conditional sampling with observed anchored via impute_mask
    """

    def __init__(self, x_dim: int, device: torch.device, cfg: Optional[DiffusionImputerConfig] = None):
        self.cfg = cfg or DiffusionImputerConfig()
        self.device = device
        self.x_dim = int(x_dim)

        self.model = EDMModel(d_in=self.x_dim, hid_dim=int(self.cfg.hidden)).to(self.device)

        # EMA shadow model (DiffPuter-style stabilization)
        self.ema_model = copy.deepcopy(self.model).to(self.device)
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    def reset(self):
        """Re-init the underlying network (optional strict EM)."""
        def _init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

        self.model.apply(_init)
        self.ema_model = copy.deepcopy(self.model).to(self.device)
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def _ema_update(self, decay: float):
        """Update EMA parameters: ema = decay*ema + (1-decay)*model."""
        if decay is None:
            return
        decay = float(decay)
        decay = max(0.0, min(0.999999, decay))

        msd = self.model.state_dict()
        esd = self.ema_model.state_dict()
        for k, v in esd.items():
            if k not in msd:
                continue
            src = msd[k]
            if not torch.is_floating_point(v):
                esd[k] = src
            else:
                esd[k] = v * decay + src * (1.0 - decay)
        self.ema_model.load_state_dict(esd, strict=False)

    def fit(self, X_complete: np.ndarray, verbose: bool = True) -> Dict[str, list]:
        """
        X_complete: (N, x_dim) standardized completed data (no NaN).
        """
        X = torch.tensor(X_complete, device=self.device, dtype=torch.float32)
        ds = TensorDataset(X)
        dl = DataLoader(ds, batch_size=int(self.cfg.batch_size), shuffle=True, drop_last=False)

        opt = torch.optim.Adam(self.model.parameters(), lr=float(self.cfg.lr))

        self.model.train()
        self.ema_model.eval()
        history = {"loss": []}

        pbar = range(int(self.cfg.epochs))
        if verbose:
            pbar = tqdm(pbar, desc="Diffusion(EDM) train", leave=False)

        for _ in pbar:
            total = 0.0
            n = 0
            for (x0,) in dl:
                loss_tensor = self.model(x0)
                loss = loss_tensor.mean()

                opt.zero_grad(set_to_none=True)
                loss.backward()
                if self.cfg.grad_clip and self.cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.cfg.grad_clip))
                opt.step()

                if self.cfg.ema is not None and float(self.cfg.ema) > 0:
                    self._ema_update(float(self.cfg.ema))

                bs = x0.shape[0]
                total += float(loss.item()) * bs
                n += bs

            history["loss"].append(total / max(n, 1))

        self.model.eval()
        self.ema_model.eval()
        return history

    @torch.no_grad()
    def impute(self, X_obs: np.ndarray, obs_mask: np.ndarray) -> np.ndarray:
        """
        X_obs: (N, x_dim) standardized; missing positions can be any number (recommend 0)
        obs_mask: (N, x_dim) with 1 observed, 0 missing

        returns: (N, x_dim) standardized imputed (MC average)
        """
        x_obs = torch.tensor(X_obs, device=self.device, dtype=torch.float32)
        obs_mask_t = torch.tensor(obs_mask, device=self.device, dtype=torch.float32)
        mask_missing = (obs_mask_t <= 0.5)

        S = int(self.cfg.num_impute_samples)
        num_steps = int(self.cfg.sample_steps)
        inner_gibbs = int(getattr(self.cfg, "inner_gibbs", 1))

        outs = []
        for _ in range(S):
            x_imp = impute_mask(
                net=self.ema_model.net,
                x_obs=x_obs,
                mask_missing=mask_missing,
                num_steps=num_steps,
                inner_gibbs=inner_gibbs,
            )
            if self.cfg.clamp_val is not None:
                x_imp = torch.clamp(x_imp, -float(self.cfg.clamp_val), float(self.cfg.clamp_val))
            outs.append(x_imp)

        x_mean = torch.stack(outs, dim=0).mean(dim=0)
        return x_mean.detach().cpu().numpy()
