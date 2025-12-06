# models/diffusion_utils.py
import torch
import numpy as np

randn_like = torch.randn_like

# hyperparameters (same as DiffPuter)
SIGMA_MIN = 0.002
SIGMA_MAX = 80
rho = 7
S_churn = 1
S_min = 0
S_max = float("inf")
S_noise = 1


# ----------------------------------------------------------
# sample_step：扩散模型单步逆扩散
# ----------------------------------------------------------
def sample_step(net, num_steps, i, t_cur, t_next, x_next):

    x_cur = x_next

    # 1. 可能增加噪声
    gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
    t_hat = net.round_sigma(t_cur + gamma * t_cur)

    x_hat = x_cur + (t_hat**2 - t_cur**2).sqrt() * S_noise * randn_like(x_cur)

    # 2. Euler step
    denoised = net(x_hat, t_hat).to(torch.float32)
    d_cur = (x_hat - denoised) / t_hat
    x_next = x_hat + (t_next - t_hat) * d_cur

    # 3. 2nd-order correction
    if i < num_steps - 1:
        denoised = net(x_next, t_next).to(torch.float32)
        d_prime = (x_next - denoised) / t_next
        x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next


# ----------------------------------------------------------
# impute_mask：混合采样填补
# ----------------------------------------------------------
def impute_mask(net, x, mask, num_samples, dim, num_steps=50, device="cpu"):

    step_indices = torch.arange(num_steps, dtype=torch.float32, device=device)

    x_t = torch.randn([num_samples, dim], device=device)

    sigma_min = max(SIGMA_MIN, net.sigma_min)
    sigma_max = min(SIGMA_MAX, net.sigma_max)

    # 构造噪声 schedule
    t_steps = (
        sigma_max ** (1 / rho)
        + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho

    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

    mask = mask.to(torch.int).to(device)
    x_t = x_t.to(torch.float32) * t_steps[0]

    N = 20  # inner refinement steps

    with torch.no_grad():
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):

            if i < num_steps - 1:

                for j in range(N):

                    n_prev = torch.randn_like(x_t) * t_next

                    x_known_t_prev = x + n_prev
                    x_unknown_t_prev = sample_step(
                        net, num_steps, i, t_cur, t_next, x_t
                    )

                    x_t_prev = x_known_t_prev * (1 - mask) + x_unknown_t_prev * mask

                    if j == N - 1:
                        x_t = x_t_prev
                    else:
                        # add noise for refinement
                        x_t = x_t_prev + torch.randn_like(x_t) * (t_cur**2 - t_next**2).sqrt()

            else:
                # 最后一层：直接设定为混合结果
                n_prev = torch.randn_like(x_t) * t_next
                x_known_t_prev = x + n_prev
                x_unknown_t_prev = sample_step(net, num_steps, i, t_cur, t_next, x_t)

                x_t = x_known_t_prev * (1 - mask) + x_unknown_t_prev * mask

    return x_t
