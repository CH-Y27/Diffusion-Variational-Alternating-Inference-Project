# models/diffusion_utils.py
import numpy as np
import torch

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
