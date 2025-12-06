# vb/nagvac.py
"""
NAGVAC (practical version):
使用因子协方差高斯近似 + PyTorch 自动求导做固定形式 VB。

q(θ) = N(μ, Σ),  Σ = b b^T + diag(c^2)

与论文中的 NAGVAC 使用同一族近似（μ, b, c），
但不手工实现自然梯度，而是用 Adam 直接最大化 ELBO：
    ELBO = E_q[ log p(θ, y | X) - log q(θ) ]
"""

from dataclasses import dataclass
from typing import List

import torch
from torch import nn
from torch.distributions import MultivariateNormal
import numpy as np
from tqdm import trange

from configs import nagvac_config
from tasks.linear_regression import BayesianLinearRegression


@dataclass
class VBResult:
    mu: np.ndarray
    cov: np.ndarray
    elbo_history: List[float]


class NAGVAC(nn.Module):
    """
    因子协方差高斯变分：
        q(θ) = N(μ, Σ),  Σ = b b^T + diag(c^2)

    重参数化采样：
        θ = μ + ε1 b + c ⊙ ε2
        ε1 ~ N(0, 1), ε2 ~ N_d(0, I)
    """

    def __init__(self, dim: int,
                 cfg=nagvac_config,
                 device: torch.device | None = None):
        super().__init__()
        self.dim = dim
        self.cfg = cfg
        self.device = device or torch.device("cpu")

        # 变分均值 μ
        self.mu = nn.Parameter(torch.zeros(dim, device=self.device))
        # 因子向量 b
        self.b = nn.Parameter(0.1 * torch.randn(dim, device=self.device))
        # c 使用 log 参数化，保证正
        self.log_c = nn.Parameter(torch.zeros(dim, device=self.device))

    # ---- 内部工具函数 ---- #

    def get_c(self) -> torch.Tensor:
        # softplus 保证 c > 0，避免协方差奇异
        return torch.nn.functional.softplus(self.log_c) + 1e-4

    def get_cov(self) -> torch.Tensor:
        """
        Σ = b b^T + diag(c^2)
        """
        b = self.b
        c = self.get_c()
        cov = b.unsqueeze(1) @ b.unsqueeze(0)  # b b^T
        cov = cov + torch.diag(c * c)
        return cov

    def sample_theta(self, n_samples: int) -> torch.Tensor:
        """
        采样 θ ~ q(θ):
            θ = μ + ε1 b + c ⊙ ε2
        返回形状 [S, D]
        """
        S = n_samples
        D = self.dim
        mu = self.mu
        b = self.b
        c = self.get_c()

        eps1 = torch.randn(S, 1, device=self.device)      # [S,1]
        eps2 = torch.randn(S, D, device=self.device)      # [S,D]
        # broadcasting：eps1*b => [S,D]
        theta = mu + eps1 * b + eps2 * c
        return theta

    def log_q(self, theta: torch.Tensor) -> torch.Tensor:
        """
        计算 log q(θ)，theta 形状 [S, D]
        """
        cov = self.get_cov()
        mvn = MultivariateNormal(self.mu, covariance_matrix=cov)
        return mvn.log_prob(theta)  # [S]

    # ---- 训练主函数 ---- #

    def fit(self, model: BayesianLinearRegression) -> VBResult:
        self.to(self.device)
        cfg = self.cfg

        optimizer = torch.optim.Adam(self.parameters(), lr=cfg.lr)

        elbo_history: list[float] = []

        for it in trange(cfg.n_iter, desc="VB (NAGVAC-factor)"):
            optimizer.zero_grad()

            # S 个样本
           # theta_samples = self.sample_theta(cfg.mc_samples)   # [S, D]

            # log p(θ,y|X)
            # 注意：linear_regression.log_joint 接受 [D,S]
            #log_joint = model.log_joint(theta_samples.T)        # [S]

            #注释前面几行后修改为：
            theta_samples = self.sample_theta(cfg.mc_samples)  # [S, D]
            log_joint = model.log_joint(theta_samples)  # [S]

            log_q = self.log_q(theta_samples)                   # [S]

            elbo = (log_joint - log_q).mean()
            loss = -elbo
            loss.backward()
            optimizer.step()

            elbo_history.append(elbo.item())

        # 提取 μ, Σ
        with torch.no_grad():
            cov = self.get_cov().detach().cpu().numpy()
            mu = self.mu.detach().cpu().numpy()

        return VBResult(mu=mu, cov=cov, elbo_history=elbo_history)
