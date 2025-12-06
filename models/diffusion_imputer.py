# models/diffusion_imputer.py
import numpy as np
import torch
import torch.nn as nn

from .diffusion_model import MLPDiffusion, Precond, EDMLoss

randn_like = torch.randn_like

# --------- 扩散超参：适当减小噪声，避免数值爆炸 ----------
SIGMA_MIN = 0.002
SIGMA_MAX = 10.0   # 原来是 80.0，过大；在标准化后取 5~10 更稳
rho = 7
S_churn = 1
S_min = 0
S_max = float("inf")
S_noise = 1


# ----------------------------------------------------------
# 单步采样：从 x_t 到 x_{t-1}
# ----------------------------------------------------------
def sample_step(net, num_steps, i, t_cur, t_next, x_next):
    x_cur = x_next

    # t_cur 这里是 0-dim tensor，先取出标量再做判断，避免潜在歧义
    t_cur_val = float(t_cur.detach().cpu().item()) if torch.is_tensor(t_cur) else float(t_cur)
    if S_min <= t_cur_val <= S_max:
        gamma = min(S_churn / num_steps, np.sqrt(2.0) - 1.0)
    else:
        gamma = 0.0

    t_hat = net.round_sigma(t_cur + gamma * t_cur).to(x_cur.device)

    # 1) 暂时增加噪声
    x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

    # 2) Euler step
    denoised = net(x_hat, t_hat)
    d_cur = (x_hat - denoised) / t_hat
    x_next = x_hat + (t_next - t_hat) * d_cur

    # 3) 2阶修正
    if i < num_steps - 1:
        denoised2 = net(x_next, t_next)
        d_prime = (x_next - denoised2) / t_next
        x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next


# ----------------------------------------------------------
# 混合采样填补 impute_mask
#   - 已观测：x 部分直接加噪后代入
#   - 缺失：用扩散逆过程生成
#   注意：这里的 x 是 **已经标准化后的数据**
# ----------------------------------------------------------
def impute_mask(
    net,
    x,              # [N, D] 标准化后的当前填补值
    mask,           # [N, D], 1=缺失, 0=观测
    num_samples,
    dim,
    num_steps,
    device,
    N_inner=5,
):
    step_indices = torch.arange(num_steps, dtype=torch.float32, device=device)
    x_t = torch.randn([num_samples, dim], device=device)

    sigma_min = max(SIGMA_MIN, net.sigma_min)
    sigma_max = min(SIGMA_MAX, net.sigma_max)

    # 噪声 schedule（EDM 形式）
    t_steps = (
        sigma_max ** (1.0 / rho)
        + step_indices / (num_steps - 1) * (sigma_min ** (1.0 / rho) - sigma_max ** (1.0 / rho))
    ) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]).to(device)

    mask = mask.to(torch.int).to(device)
    x_t = x_t * t_steps[0]

    with torch.no_grad():
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            for j in range(N_inner):
                n_prev = torch.randn_like(x_t) * t_next

                # 已观测位置：当前 x + 噪声
                x_known = x + n_prev
                # 缺失位置：走一次采样步
                x_unknown = sample_step(net, num_steps, i, t_cur, t_next, x_t)

                x_t_prev = x_known * (1 - mask) + x_unknown * mask

                if j == N_inner - 1:
                    x_t = x_t_prev
                else:
                    x_t = x_t_prev + torch.randn_like(x_t) * (t_cur ** 2 - t_next ** 2).sqrt()

    return x_t


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
                N_inner=5,
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
