# vb/cholesky_gvb.py
from dataclasses import dataclass
from typing import Dict, Any
import torch
from torch import nn
from torch.distributions import MultivariateNormal
from tqdm import trange
import numpy as np

from configs import vb_config
from tasks.linear_regression import BayesianLinearRegression

@dataclass
class VBResult:
    mu: np.ndarray
    cov: np.ndarray
    elbo_history: list[float]

class CholeskyGVB(nn.Module):
    """
    全协方差高斯变分推断：
        q(θ) = N(μ, Σ), Σ = L L^T,  L 为下三角
    使用重参数化梯度 + Adam 直接优化 ELBO
    """

    def __init__(self, dim: int, cfg=vb_config, device=None):
        super().__init__()
        self.dim = dim
        self.cfg = cfg
        self.device = device or torch.device("cpu")

        # 变分均值 μ
        self.mu = nn.Parameter(torch.zeros(dim, device=self.device))

        # Cholesky 下三角参数：先构造一个接近对角阵的 L
        init_L = cfg.init_scale * torch.eye(dim)
        self.L_raw = nn.Parameter(init_L.to(self.device))

    def get_L(self) -> torch.Tensor:
        # 强制为下三角，并保证对角线为正（用 softplus）
        L = torch.tril(self.L_raw)
        diag = torch.diag(torch.nn.functional.softplus(torch.diag(L)) + 1e-4)
        L = L - torch.diag(torch.diag(L)) + diag
        return L

    def sample_theta(self, n_samples: int) -> torch.Tensor:
        """
        重参数化采样 θ ~ q(θ)
        返回形状 [n_samples, dim]
        """
        L = self.get_L()
        eps = torch.randn(n_samples, self.dim, device=self.device)
        theta = self.mu + eps @ L.T
        return theta

    def log_q(self, theta: torch.Tensor) -> torch.Tensor:
        L = self.get_L()
        cov = L @ L.T
        mvn = MultivariateNormal(self.mu, covariance_matrix=cov)
        return mvn.log_prob(theta)

    def fit(self, model: BayesianLinearRegression) -> VBResult:
        self.to(self.device)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.lr)

        elbo_history: list[float] = []

        for it in trange(self.cfg.n_iter, desc="VB (Cholesky GVB)"):
            optimizer.zero_grad()

            # S 个 MC 样本
            theta_samples = self.sample_theta(self.cfg.mc_samples)  # [S, D]

            log_joint = model.log_joint(theta_samples.T)    # [S]
            log_q = self.log_q(theta_samples)               # [S]
            elbo = (log_joint - log_q).mean()

            loss = -elbo  # 最大化 ELBO == 最小化负号
            loss.backward()
            optimizer.step()

            elbo_history.append(elbo.item())

        # 提取最终 μ 和 Σ
        L = self.get_L().detach().cpu()
        mu = self.mu.detach().cpu().numpy()
        cov = (L @ L.T).numpy()

        return VBResult(mu=mu, cov=cov, elbo_history=elbo_history)
