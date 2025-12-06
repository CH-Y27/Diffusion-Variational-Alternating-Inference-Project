# models/diffusion_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ----------------------------
# 基础激活 & 嵌入
# ----------------------------
class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class PositionalEmbedding(nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        # x: [B] 标量时间 / 噪声标签
        freqs = torch.arange(
            start=0,
            end=self.num_channels // 2,
            dtype=torch.float32,
            device=x.device,
        )
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


class MLPDiffusion(nn.Module):
    """
    一个简单的 MLP 去噪网络，用于 EDM 框架。
    输入: x [B, d_in], noise_labels [B] (标量, log-sigma 或 sigma 的某种函数)
    输出: 与 x 同维度的向量
    """

    def __init__(self, d_in: int, dim_t: int = 512):
        super().__init__()
        self.dim_t = dim_t

        self.proj = nn.Linear(d_in, dim_t)

        self.mlp = nn.Sequential(
            nn.Linear(dim_t, dim_t * 2),
            nn.SiLU(),
            nn.Linear(dim_t * 2, dim_t * 2),
            nn.SiLU(),
            nn.Linear(dim_t * 2, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, d_in),
        )

        self.map_noise = PositionalEmbedding(num_channels=dim_t)
        self.time_embed = nn.Sequential(
            nn.Linear(dim_t, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, dim_t),
        )

    def forward(self, x, noise_labels):
        """
        x: [B, d_in]
        noise_labels: [B] 标量 (比如 log sigma)
        """
        # 噪声标签嵌入
        if noise_labels.dim() == 1:
            n = noise_labels
        else:
            n = noise_labels.view(-1)

        emb = self.map_noise(n)  # [B, dim_t]
        # Flip sin/cos 顺序
        emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape)
        emb = self.time_embed(emb)

        x = self.proj(x)
        x = x + emb
        return self.mlp(x)


class Precond(nn.Module):
    """
    与 DiffPuter / EDM 一致的 preconditioning 包装。
    """

    def __init__(
        self,
        denoise_fn: nn.Module,
        hid_dim: int,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        sigma_data: float = 0.5,
    ):
        super().__init__()
        self.hid_dim = hid_dim
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data

        self.denoise_fn_F = denoise_fn  # 真正的 MLP 去噪网络

    def forward(self, x, sigma):
        """
        x: [B, d]
        sigma: [B] 或 [B, 1]
        返回 D(x, sigma), 与 EDM 论文一致
        """
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1)
        dtype = torch.float32

        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()
        c_in = 1 / (self.sigma_data**2 + sigma**2).sqrt()
        c_noise = sigma.log() / 4

        x_in = c_in * x
        F_x = self.denoise_fn_F(x_in.to(dtype), c_noise.view(-1))

        assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x

    def round_sigma(self, sigma):
        # EDM 里是用表格，这里直接返回即可
        return torch.as_tensor(sigma, dtype=torch.float32, device=sigma.device)


class EDMLoss:
    """
    EDM 论文中的 loss（简化版），与 DiffPuter 一致。
    """

    def __init__(
        self,
        P_mean: float = -1.2,
        P_std: float = 1.2,
        sigma_data: float = 0.5,
        hid_dim: int = 100,
        gamma: float = 5,
        opts=None,
    ):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.hid_dim = hid_dim
        self.gamma = gamma
        self.opts = opts

    def __call__(self, denoise_fn: nn.Module, data: torch.Tensor):
        """
        denoise_fn: 一般就是 Precond 对象
        data: [B, d]
        返回 per-sample loss [B]
        """
        device = data.device
        B = data.shape[0]

        rnd_normal = torch.randn(B, device=device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()  # [B]

        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2

        y = data
        n = torch.randn_like(y) * sigma.unsqueeze(1)

        D_yn = denoise_fn(y + n, sigma)  # [B, d]，Precond forward

        target = y
        loss = weight.unsqueeze(1) * (D_yn - target) ** 2  # [B, d]
        # 返回 [B]，与 DiffPuter 一致（外面再 mean）
        return loss.mean(dim=1)
