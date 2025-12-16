# models/diffusion_utils.py
import numpy as np
import torch

randn_like = torch.randn_like

# --------- 扩散超参：完全对齐 DIFFPUTER 官方实现 ----------
SIGMA_MIN = 0.002
SIGMA_MAX = 80.0    # 恢复为 80
rho = 7.0
S_churn = 1.0
S_min = 0.0
S_max = float("inf")
S_noise = 1.0


# ----------------------------------------------------------
# 单步采样：从 x_t 到 x_{t-1}（EDM 采样器）
# ----------------------------------------------------------
def sample_step(net, num_steps, i, t_cur, t_next, x_next):
    """
    net:  预条件后的网络 (Precond)，具有 round_sigma / sigma_min / sigma_max 属性
    t_cur, t_next: 标量张量，对应当前与下一步的 sigma
    x_next: 当前 x_t
    """
    x_cur = x_next

    # 暂时增加噪声（Karras S_churn）
    gamma = min(S_churn / num_steps, np.sqrt(2.0) - 1.0) if (S_min <= t_cur <= S_max) else 0.0
    t_hat = net.round_sigma(t_cur + gamma * t_cur)

    # 1) 提高噪声
    x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

    # 2) Euler 步
    denoised = net(x_hat, t_hat).to(torch.float32)
    d_cur = (x_hat - denoised) / t_hat
    x_next = x_hat + (t_next - t_hat) * d_cur

    # 3) 2 阶修正（Heun）
    if i < num_steps - 1:
        denoised2 = net(x_next, t_next).to(torch.float32)
        d_prime = (x_next - denoised2) / t_next
        x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next


# ----------------------------------------------------------
# 条件采样 + 掩码混合（DIFFPUTER 的 impute_mask）
#   x    : [N, D] 标准化后的当前填补数据
#   mask : [N, D]，1=缺失, 0=观测
#   返回 : x_t≈0 时的一条条件样本（标准化空间）
# ----------------------------------------------------------
@torch.no_grad()
def impute_mask(
    net,
    x,
    mask,
    num_samples,
    dim,
    num_steps=50,
    device="cpu",
    N_inner=20,
):
    """
    net        : Precond(MLPDiffusion)
    x          : (num_samples, dim) 当前填补后的数据（标准化）
    mask       : (num_samples, dim) 掩码，1=missing, 0=observed
    num_steps  : 时间步 M
    N_inner    : 每个时间步内部的随机小步数（原论文为 20）
    """
    step_indices = torch.arange(num_steps, dtype=torch.float32, device=device)
    x_t = torch.randn([num_samples, dim], device=device)

    # 与网络自身的 sigma 区间取交集
    sigma_min = max(SIGMA_MIN, float(net.sigma_min))
    sigma_max = min(SIGMA_MAX, float(net.sigma_max))

    # Karras sigma schedule
    t_steps = (
        sigma_max ** (1.0 / rho)
        + step_indices / (num_steps - 1)
        * (sigma_min ** (1.0 / rho) - sigma_max ** (1.0 / rho))
    ) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

    x = x.to(torch.float32).to(device)
    mask = mask.to(torch.int).to(device)
    x_t = x_t.to(torch.float32) * t_steps[0]

    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        if i < num_steps - 1:
            # 非最后一层：做 N_inner 个随机小步
            for j in range(N_inner):
                # 观测维前向扰动
                n_prev = torch.randn_like(x_t) * t_next
                x_known_t_prev = x + n_prev

                # 缺失维逆向采样
                x_unknown_t_prev = sample_step(net, num_steps, i, t_cur, t_next, x_t)

                # 掩码融合
                x_t_prev = x_known_t_prev * (1 - mask) + x_unknown_t_prev * mask

                if j == N_inner - 1:
                    x_t = x_t_prev
                else:
                    # 额外随机扰动，近似积分 t_cur -> t_next
                    n = torch.randn_like(x_t) * (t_cur.pow(2) - t_next.pow(2)).sqrt()
                    x_t = x_t_prev + n
        else:
            # 最后一层：只做一次 forward+reverse+mask，不再加随机噪声
            n_prev = torch.randn_like(x_t) * t_next
            x_known_t_prev = x + n_prev
            x_unknown_t_prev = sample_step(net, num_steps, i, t_cur, t_next, x_t)

            x_t = x_known_t_prev * (1 - mask) + x_unknown_t_prev * mask

    return x_t
