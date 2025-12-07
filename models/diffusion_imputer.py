# models/diffusion_imputer.py
import numpy as np
import torch
import torch.nn as nn

from .diffusion_model import MLPDiffusion, Precond, EDMLoss
from .diffusion_utils import impute_mask   # ⭐ 关键改动：从 utils 导入

# ----------------------------------------------------------
# 简化版 DVAI 用扩散填补器（带标准化 & 数值保护）
# ----------------------------------------------------------
class SimpleDiffusionImputer:
    """
    - 训练阶段：用 EDM loss 在当前 "填补后 X" 上训练 score 网络
    - 采样阶段：在标准化空间用 impute_mask 在缺失位置生成新值
    - 内部自动维护列均值/方差用于标准化，外部接口始终用原尺度的 X
    """

    def __init__(self, dim_x, hidden_dim=256, num_steps=30, J=3, device="cpu"):
        self.dim_x = dim_x
        self.hidden_dim = hidden_dim
        self.device = device
        self.num_steps = num_steps
        self.J = J

        self.denoise_fn = MLPDiffusion(d_in=dim_x, dim_t=hidden_dim).to(device)
        self.precond = Precond(self.denoise_fn, hidden_dim).to(device)
        self.loss_fn = EDMLoss()
        self.opt = torch.optim.Adam(self.precond.parameters(), lr=1e-4)

        # 标准化所需的统计量
        self.mean_ = None
        self.std_ = None

    # ---------------- 内部：拟合 & 使用 scaler ----------------
    def _fit_scaler(self, X: np.ndarray):
        """
        对当前填补后的 X 拟合列均值和标准差
        """
        mean = np.nanmean(X, axis=0)
        std = np.nanstd(X, axis=0)
        std[std < 1e-3] = 1.0   # 避免过小方差引起放大

        self.mean_ = mean
        self.std_ = std

    def _transform(self, X: np.ndarray) -> np.ndarray:
        """
        映射到标准化空间，大致在 [-1, 1] 范围
        """
        if self.mean_ is None or self.std_ is None:
            self._fit_scaler(X)
        return (X - self.mean_) / (self.std_ * 2.0)

    def _inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        """
        从标准化空间映回原尺度
        """
        return Z * (self.std_ * 2.0) + self.mean_

    # ---------------- 训练：在标准化空间上训练 EDM ----------------
    def train_model(self, X: np.ndarray, iters: int = 80):
        """
        X: 当前填补后的数据 (N, D)，原尺度
        """
        # 更新标准化参数，并将 X 映射到标准化空间
        Z = self._transform(X)      # (N, D) 约在 [-1, 1]
        Z_t = torch.tensor(Z, dtype=torch.float32, device=self.device)

        for k in range(iters):
            loss_vec = self.loss_fn(self.precond, Z_t)
            loss = loss_vec.mean()
            if not torch.isfinite(loss):
                print(f"[Diffusion][Warn] loss 非有限（{loss.item()}），提前停止本轮训练")
                break

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            if k % 20 == 0:
                print(f"[Diffusion] step {k}/{iters}, loss={loss.item():.4f}")

    # ---------------- 采样：在标准化空间填补，再映回原尺度 ----------------
    @torch.no_grad()
    def impute(self, X_current: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        X_current: (N, D) 当前一轮的填补数据（原尺度）
        mask:      (N, D) 1=缺失, 0=观测
        返回: 新一轮填补后的 X_new（原尺度）
        """
        # 用当前 scaler 把输入映射到标准化空间
        Z = self._transform(X_current)     # (N, D)
        Z_t = torch.tensor(Z, dtype=torch.float32, device=self.device)
        mask_t = torch.tensor(mask, dtype=torch.int, device=self.device)

        N, D = Z_t.shape
        samples = []

        for j in range(self.J):
            Z_j = impute_mask(
                self.precond,
                Z_t,
                mask_t,
                num_samples=N,
                dim=D,
                num_steps=self.num_steps,
                device=self.device,
                N_inner=5,   # 仍然使用你之前设定的 inner 步数
            )
            samples.append(Z_j)

        Z_mean = torch.stack(samples).mean(0).cpu().numpy()
        X_new = self._inverse_transform(Z_mean)

        # --------- 数值保护：处理 NaN/Inf + 截断 ----------
        bad = ~np.isfinite(X_new)
        if bad.any():
            num_bad = bad.sum()
            print(f"[Diffusion][Warn] impute 产生 {num_bad} 个非有限值，回退到上一轮 X")
            X_new[bad] = X_current[bad]

        # 软截断，避免极端大值污染 VB
        X_new = np.clip(X_new, -10.0, 10.0)

        return X_new
