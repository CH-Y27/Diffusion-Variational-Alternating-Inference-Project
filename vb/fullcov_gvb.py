# vb/fullcov_gvb.py
"""
Full-covariance Gaussian Variational Bayes (GVB-full)

q(theta) = N(mu, Sigma), Sigma = L L^T  (L lower-triangular)

在当前线性任务中：
  - 真后验本身就是高斯；
  - full-cov GVB 优化 ELBO 的最优解就是精确后验；
  - 不依赖共轭解析式，只用 log_joint(theta) 即可，
    因此后续可以迁移到非共轭任务（如 logistic + 扩散）。

接口要求 model 提供：
    log_joint(theta) -> Tensor[S]  （theta: [S, D]）
"""

from dataclasses import dataclass
from typing import List

import torch
from torch import nn
import numpy as np
from tqdm import trange


@dataclass
class FullCovVBConfig:
    n_iter: int = 10000          # 最大迭代次数
    mc_samples: int = 20         # 每步 MC 样本数 S
    lr: float = 1e-3             # Adam 学习率
    grad_clip: float = 50.0      # 梯度裁剪阈值
    lb_smooth: int = 50          # ELBO 平滑窗口长度
    patience: int = 500          # 早停耐心（以平滑 ELBO 计）


class FullCovGaussianVB(nn.Module):
    """
    全协方差高斯变分推断（GVB-full）

    参数：
        dim: 参数维度 D
        cfg: FullCovVBConfig
        device: torch.device

    注意：
        这里我们不手动写自然梯度，而是使用重参数化 + Adam。
        在线性高斯模型中，ELBO 是凸的，只要优化收敛，
        GVB-full 的最优解 = 精确后验（与 NAGVAC 的最优解相同）。
    """

    def __init__(self, dim: int, cfg: FullCovVBConfig, device: torch.device):
        super().__init__()
        self.dim = dim
        self.cfg = cfg
        self.device = device

        # 变分均值 μ
        self.mu = nn.Parameter(torch.zeros(dim, device=device))

        # 下三角矩阵 L 的原始参数（用一个任意矩阵，然后取 tril 并让对角线为正）
        L_init = torch.eye(dim, device=device)
        self.L_unconstrained = nn.Parameter(L_init)

    def _get_L(self) -> torch.Tensor:
        """
        从 L_unconstrained 得到真正的下三角矩阵 L，
        对角线用 softplus 保证为正，从而 Sigma = L L^T 一定正定。
        """
        L = torch.tril(self.L_unconstrained)
        diag = torch.nn.functional.softplus(torch.diag(L)) + 1e-4
        L = L - torch.diag(torch.diag(L)) + torch.diag(diag)
        return L

    def sample_theta(self, S: int):
        """
        采样 θ_s = μ + L ε_s,  ε_s ~ N(0, I)

        返回：
            theta: [S, D]
            L:     [D, D]
        """
        L = self._get_L()
        eps = torch.randn(S, self.dim, device=self.device)
        theta = self.mu.unsqueeze(0) + eps @ L.T   # [S, D]
        return theta, L

    def log_q(self, theta: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        """
        计算 log q(theta)，theta: [S, D]
        q = N(mu, Sigma), Sigma = L L^T
        """
        D = self.dim
        # Σ^{-1} = (L L^T)^{-1} = (L^{-1})^T L^{-1}
        # 这里用 cholesky_inverse(L) 直接得到 Σ^{-1}
        inv_Sigma = torch.cholesky_inverse(L)
        diff = theta - self.mu.unsqueeze(0)       # [S, D]
        quad = torch.sum(diff @ inv_Sigma * diff, dim=1)  # [S]
        logdet = 2.0 * torch.log(torch.diag(L)).sum()     # 标量 log|Σ|
        log_q = -0.5 * (quad + logdet + D * torch.log(torch.tensor(2.0 * torch.pi, device=self.device)))
        return log_q

    def fit(self, model) -> dict:
        """
        对给定 model 进行变分推断，model 需实现：
            log_joint(theta) -> [S]  (theta: [S, D])

        返回 dict:
            {
                "mu": np.ndarray[D],
                "cov": np.ndarray[D, D],
                "elbo_history": List[float],
            }
        """
        self.to(self.device)
        cfg = self.cfg

        optimizer = torch.optim.Adam(self.parameters(), lr=cfg.lr)
        elbo_history: List[float] = []

        best_elbo = -1e30
        best_state = None
        bad_count = 0

        for it in trange(cfg.n_iter, desc="VB (FullCov-GVB)"):
            optimizer.zero_grad()

            # 1. 采样 θ_s ~ q(θ)
            theta, L = self.sample_theta(cfg.mc_samples)   # [S, D], [D, D]

            # 2. 计算 ELBO = E_q[log p(θ,y) - log q(θ)]
            log_joint = model.log_joint(theta)             # [S]
            log_q = self.log_q(theta, L)                   # [S]
            elbo = (log_joint - log_q).mean()
            loss = -elbo

            loss.backward()

            # 3. 梯度裁剪，避免数值爆炸
            total_norm = 0.0
            for p in self.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm().item() ** 2
            total_norm = total_norm ** 0.5
            if total_norm > cfg.grad_clip:
                clip_coef = cfg.grad_clip / (total_norm + 1e-8)
                for p in self.parameters():
                    if p.grad is not None:
                        p.grad.data.mul_(clip_coef)

            optimizer.step()

            elbo_val = elbo.item()
            elbo_history.append(elbo_val)

            # 4. 平滑 ELBO + early stopping
            if len(elbo_history) >= cfg.lb_smooth:
                avg_elbo = float(
                    sum(elbo_history[-cfg.lb_smooth:]) / cfg.lb_smooth
                )
            else:
                avg_elbo = elbo_val

            if avg_elbo > best_elbo:
                best_elbo = avg_elbo
                best_state = {
                    "mu": self.mu.detach().clone().cpu(),
                    "L_unconstrained": self.L_unconstrained.detach().clone().cpu(),
                }
                bad_count = 0
            else:
                bad_count += 1

            if bad_count > cfg.patience:
                break

        # 恢复最佳状态
        if best_state is not None:
            self.mu.data = best_state["mu"].to(self.device)
            self.L_unconstrained.data = best_state["L_unconstrained"].to(self.device)

        # 提取最终 μ 和 Σ
        with torch.no_grad():
            L_final = self._get_L()
            Sigma_final = (L_final @ L_final.T).cpu().numpy()
            mu_final = self.mu.detach().cpu().numpy()

        return {
            "mu": mu_final,
            "cov": Sigma_final,
            "elbo_history": elbo_history,
        }
