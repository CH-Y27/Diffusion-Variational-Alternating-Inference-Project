# models/diffusion_imputer.py
import numpy as np
import torch
import torch.nn as nn

from .diffusion_model import MLPDiffusion, Precond, EDMLoss
from .diffusion_utils import impute_mask   # 混合采样填补


# ----------------------------------------------------------
# DVAI 用扩散填补器（标准化 + 数值保护 + 加强采样）
# ----------------------------------------------------------
class SimpleDiffusionImputer:
    """
    - 训练阶段：用 EDM loss 在当前 "填补后 X" 上训练 score 网络
    - 采样阶段：在标准化空间用 impute_mask 在缺失位置生成新值
    - 内部自动维护列均值/方差用于标准化，外部接口始终用原尺度的 X
    """

    def __init__(
        self,
        dim_x: int,
        hidden_dim: int = 512,   # 加宽网络
        num_steps: int = 50,     # 逆过程时间步数
        J: int = 10,             # 条件采样次数
        device: str = "cpu",
    ):
        self.dim_x = dim_x
        self.hidden_dim = hidden_dim
        self.device = device
        self.num_steps = num_steps
        self.J = J

        # 生成网络 + 预条件封装
        self.denoise_fn = MLPDiffusion(d_in=dim_x, dim_t=hidden_dim).to(device)
        self.precond = Precond(self.denoise_fn, hidden_dim).to(device)
        self.loss_fn = EDMLoss()
        self.opt = torch.optim.Adam(self.precond.parameters(), lr=2e-4)

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
        # 避免过小方差造成数值放大
        std[std < 1e-3] = 1.0

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
    def train_model(self, X: np.ndarray, iters: int = 150):
        """
        X: 当前填补后的数据 (N, D)，原尺度
        - 每一轮都会重新拟合标准化参数，使 scaler 适应最新填补数据
        """
        # 更新标准化参数，并将 X 映射到标准化空间
        self._fit_scaler(X)
        Z = self._transform(X)          # (N, D) 约在 [-1, 1]
        Z_t = torch.tensor(Z, dtype=torch.float32, device=self.device)

        self.precond.train()
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

        # J 次独立条件采样，最后取均值近似 E[x | x_obs]
        for j in range(self.J):
            Z_sample = impute_mask(
                net=self.precond,
                x=Z_t,
                mask=mask_t,
                num_samples=N,
                dim=D,
                num_steps=self.num_steps,
                device=self.device,
                N_inner=20,   # 比之前更细的 inner step
            )
            samples.append(Z_sample)

        Z_mean = torch.stack(samples).mean(0).cpu().numpy()
        X_new = self._inverse_transform(Z_mean)

        # --------- 数值保护：处理 NaN/Inf ----------
        bad = ~np.isfinite(X_new)
        if bad.any():
            num_bad = bad.sum()
            print(f"[Diffusion][Warn] impute 产生 {num_bad} 个非有限值，回退到上一轮 X")
            X_new[bad] = X_current[bad]

        # 软截断：根据列标准差设一个较宽的范围，避免极端大值
        if self.std_ is not None:
            max_std = float(np.max(self.std_))
            bound = 6.0 * max_std   # 约 6σ
            X_new = np.clip(X_new, -bound, bound)

        return X_new
