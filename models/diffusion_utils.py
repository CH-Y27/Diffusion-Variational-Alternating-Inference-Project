# models/diffusion_utils.py
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn

randn_like = torch.randn_like

SIGMA_MIN = 0.002
SIGMA_MAX = 80.0
RHO = 7.0


class EDMLoss:
    def __init__(self, P_mean: float = -1.2, P_std: float = 1.2, sigma_data: float = 0.5):
        self.P_mean = float(P_mean)
        self.P_std = float(P_std)
        self.sigma_data = float(sigma_data)

    def __call__(self, denoise_fn: nn.Module, data: torch.Tensor) -> torch.Tensor:
        rnd = torch.randn(data.shape[0], device=data.device)
        sigma = (rnd * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y = data
        n = torch.randn_like(y) * sigma.unsqueeze(1)
        D_yn = denoise_fn(y + n, sigma)
        loss = weight.unsqueeze(1) * (D_yn - y) ** 2
        return loss


def _make_t_steps(net, num_steps: int, device: torch.device) -> torch.Tensor:
    step_idx = torch.arange(num_steps, dtype=torch.float32, device=device)
    sigma_min = max(SIGMA_MIN, getattr(net, "sigma_min", SIGMA_MIN))
    sigma_max = min(SIGMA_MAX, getattr(net, "sigma_max", SIGMA_MAX))
    t = (sigma_max ** (1 / RHO) + step_idx / (num_steps - 1) * (sigma_min ** (1 / RHO) - sigma_max ** (1 / RHO))) ** RHO
    t = torch.cat([net.round_sigma(t), torch.zeros_like(t[:1])])
    return t


def sample_step(
    net: nn.Module,
    num_steps: int,
    i: int,
    t_cur: torch.Tensor,
    t_next: torch.Tensor,
    x_next: torch.Tensor,
    S_churn: float = 0.0,
    S_noise: float = 1.0,
) -> torch.Tensor:
    x_cur = x_next
    gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_churn > 0 else 0.0
    t_hat = net.round_sigma(t_cur + gamma * t_cur)
    x_hat = x_cur + torch.sqrt(torch.clamp(t_hat ** 2 - t_cur ** 2, min=0.0)) * S_noise * randn_like(x_cur)

    denoised = net(x_hat, t_hat).to(torch.float32)
    d_cur = (x_hat - denoised) / t_hat
    x_next = x_hat + (t_next - t_hat) * d_cur

    if i < num_steps - 1:
        denoised_2 = net(x_next, t_next).to(torch.float32)
        d_prime = (x_next - denoised_2) / t_next
        x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next


@torch.no_grad()
def impute_mask(
    net: nn.Module,
    x_obs: torch.Tensor,
    mask_missing: torch.Tensor,
    num_steps: int = 50,
    inner_gibbs: int = 1,
) -> torch.Tensor:
    device = x_obs.device
    N, D = x_obs.shape

    t_steps = _make_t_steps(net, num_steps, device)
    x_t = torch.randn((N, D), device=device, dtype=torch.float32) * t_steps[0]

    mask = mask_missing.to(device=device)
    obs_mask = (~mask).to(torch.float32)

    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        for j in range(inner_gibbs):
            n_prev = torch.randn_like(x_t) * t_next
            x_known_t_prev = x_obs + n_prev
            x_unknown_t_prev = sample_step(net, num_steps, i, t_cur, t_next, x_t)

            x_t_prev = x_known_t_prev * obs_mask + x_unknown_t_prev * mask.to(torch.float32)

            if j == inner_gibbs - 1:
                x_t = x_t_prev
            else:
                n = torch.randn_like(x_t) * torch.sqrt(torch.clamp(t_cur ** 2 - t_next ** 2, min=0.0))
                x_t = x_t_prev + n

    return x_t
