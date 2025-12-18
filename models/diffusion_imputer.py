# models/diffusion_imputer.py
# Train + sample DDPM for tabular imputation with inpainting constraints.
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .diffusion_model import MLPDenoiser
from .diffusion_utils import DiffusionSchedule, apply_inpainting_constraint


@dataclass
class DiffusionImputerConfig:
    T: int = 500                 # fewer steps -> faster CPU
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    hidden: int = 256
    n_layers: int = 4
    t_embed_dim: int = 128
    dropout: float = 0.0

    # training
    lr: float = 2e-4
    batch_size: int = 256
    epochs: int = 10
    grad_clip: float = 1.0
    ema: float = 0.999

    # sampling
    sample_steps: int = 200      # <= T; using a subset speeds up
    num_impute_samples: int = 8  # Monte Carlo samples per row
    clamp_val: float = 6.0       # keep values bounded in standardized space


class EMA:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {}
        for k, v in model.state_dict().items():
            self.shadow[k] = v.detach().clone()

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)

    @torch.no_grad()
    def copy_to(self, model: nn.Module):
        model.load_state_dict(self.shadow, strict=True)


class DiffusionImputer:
    """
    Unconditional DDPM on completed X, used as an imputer via inpainting during sampling.
    """

    def __init__(self, x_dim: int, device: torch.device, cfg: Optional[DiffusionImputerConfig] = None):
        self.cfg = cfg or DiffusionImputerConfig()
        self.device = device

        self.schedule = DiffusionSchedule(
            T=self.cfg.T,
            beta_start=self.cfg.beta_start,
            beta_end=self.cfg.beta_end,
            device=device,
            dtype=torch.float32,
        )

        self.net = MLPDenoiser(
            x_dim=x_dim,
            hidden=self.cfg.hidden,
            n_layers=self.cfg.n_layers,
            t_embed_dim=self.cfg.t_embed_dim,
            dropout=self.cfg.dropout,
        ).to(device)

        self.ema = EMA(self.net, decay=self.cfg.ema)

    def fit(self, X_complete: np.ndarray, verbose: bool = True) -> Dict[str, list]:
        """
        Train epsilon-predictor on the current completed dataset (standardized recommended).
        """
        X = torch.tensor(X_complete, device=self.device, dtype=torch.float32)
        ds = TensorDataset(X)
        dl = DataLoader(ds, batch_size=self.cfg.batch_size, shuffle=True, drop_last=True)

        opt = torch.optim.Adam(self.net.parameters(), lr=self.cfg.lr)

        history = {"loss": []}
        self.net.train()

        pbar = range(self.cfg.epochs)
        if verbose:
            pbar = tqdm(pbar, desc="Diffusion-train", leave=False)

        for _ in pbar:
            epoch_loss = 0.0
            n = 0
            for (x0,) in dl:
                B = x0.shape[0]
                t = torch.randint(0, self.schedule.T, (B,), device=self.device, dtype=torch.long)
                noise = torch.randn_like(x0)

                sa = self.schedule.sqrt_alpha_bar[t].view(-1, 1)
                so = self.schedule.sqrt_one_minus_alpha_bar[t].view(-1, 1)
                x_t = sa * x0 + so * noise

                pred = self.net(x_t, t)
                loss = torch.mean((pred - noise) ** 2)

                opt.zero_grad(set_to_none=True)
                loss.backward()
                if self.cfg.grad_clip is not None and self.cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.grad_clip)
                opt.step()
                self.ema.update(self.net)

                epoch_loss += float(loss.item()) * B
                n += B

            history["loss"].append(epoch_loss / max(n, 1))

        # use EMA weights for sampling
        self.ema.copy_to(self.net)
        self.net.eval()
        return history

    @torch.no_grad()
    def impute(self, X_obs: np.ndarray, obs_mask: np.ndarray) -> np.ndarray:
        """
        X_obs: (N, D) with any filler values at missing entries.
        obs_mask: (N, D) 1 for observed, 0 for missing.

        Returns: imputed X (N, D) using MC average over num_impute_samples.
        """
        cfg = self.cfg
        schedule = self.schedule
        net = self.net

        x0_obs = torch.tensor(X_obs, device=self.device, dtype=torch.float32)
        m = torch.tensor(obs_mask, device=self.device, dtype=torch.float32)

        N, D = x0_obs.shape
        S = cfg.num_impute_samples

        # choose a subset of timesteps for fast sampling
        steps = min(cfg.sample_steps, schedule.T)
        t_seq = torch.linspace(schedule.T - 1, 0, steps, device=self.device).long()

        out = torch.zeros((S, N, D), device=self.device, dtype=torch.float32)

        for s in range(S):
            x_t = torch.randn((N, D), device=self.device, dtype=torch.float32)

            # initial enforce at max t
            noise_obs = torch.randn_like(x_t)
            x_t = apply_inpainting_constraint(x_t, x0_obs, m, schedule, t_seq[0].expand(N), noise=noise_obs)

            for i in range(len(t_seq)):
                t = t_seq[i].expand(N)
                t_prev = t_seq[i + 1].expand(N) if i + 1 < len(t_seq) else torch.zeros_like(t)

                beta_t = schedule.betas[t].view(-1, 1)
                alpha_t = schedule.alphas[t].view(-1, 1)
                alpha_bar_t = schedule.alpha_bar[t].view(-1, 1)
                alpha_bar_prev = schedule.alpha_bar[t_prev].view(-1, 1)

                eps = net(x_t, t)

                # x0 estimate
                x0_hat = (x_t - torch.sqrt(1 - alpha_bar_t) * eps) / torch.sqrt(alpha_bar_t)
                x0_hat = torch.clamp(x0_hat, -cfg.clamp_val, cfg.clamp_val)

                # DDPM posterior mean
                coef1 = torch.sqrt(alpha_bar_prev) * beta_t / (1 - alpha_bar_t)
                coef2 = torch.sqrt(alpha_t) * (1 - alpha_bar_prev) / (1 - alpha_bar_t)
                mu = coef1 * x0_hat + coef2 * x_t

                var = schedule.posterior_variance[t].view(-1, 1)
                if i + 1 < len(t_seq):
                    z = torch.randn_like(x_t)
                    x_t = mu + torch.sqrt(torch.clamp(var, min=1e-20)) * z
                else:
                    x_t = mu  # last step (t=0)

                # enforce observed entries (inpainting) at the current t_prev level
                noise_obs = torch.randn_like(x_t)
                x_t = apply_inpainting_constraint(x_t, x0_obs, m, schedule, t_prev, noise=noise_obs)

            out[s] = x_t

        x_imp = out.mean(dim=0).detach().cpu().numpy()
        return x_imp
