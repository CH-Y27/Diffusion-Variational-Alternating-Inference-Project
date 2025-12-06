# vb/mcmc_linear.py
import numpy as np
import torch
from tqdm import trange


def run_mcmc_linear(model, cfg, seed: int = 42):
    """
    自适应随机游走 Metropolis，用于线性高斯模型的后验采样。

    参数
    ----
    model : BayesianLinearRegression
        必须实现 log_joint(theta) -> [S]，其中 theta: [S, D] (torch.Tensor)
    cfg : MCMCConfig
        configs.py 中定义的配置
    seed : int
        随机种子

    返回
    ----
    samples : np.ndarray, shape [n_eff, D]
        Burn-in 之后、thinning 之后的后验样本
    """

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = model.X.device
    D = model.D  # 在线性模型中我们在类里定义过 self.D

    n_iter = cfg.n_iter
    burn_in = int(cfg.burn_in_rate * n_iter)
    target_accept = cfg.target_accept
    num_cov = cfg.num_covariance
    sig_scale = cfg.sig_scale
    scale = cfg.init_scale
    thin = cfg.thin
    verbose = cfg.verbose

    # --- 初始化 ---
    # 线性模型下：先从先验采一个初值，或者直接用 0 向量也可以
    theta = torch.zeros(D, device=device)
    log_post = model.log_joint(theta.unsqueeze(0))[0].item()

    # 初始 proposal 协方差 V
    V = sig_scale * np.eye(D, dtype=np.float64)

    # 存储所有 θ（包括 burn-in）
    all_samples = np.zeros((n_iter, D), dtype=np.float64)

    accept_count = 0

    # 为了估计协方差，单独存最近样本的缓冲区
    window_samples = []

    # --- MCMC 主循环 ---
    for it in trange(n_iter, desc="MCMC"):
        # 1) 在当前 θ 附近提出 proposal
        #    θ* = θ + scale * L * eps
        L = np.linalg.cholesky(V + 1e-8 * np.eye(D))
        eps = np.random.randn(D)
        delta = scale * (L @ eps)
        theta_prop = theta + torch.tensor(delta, dtype=torch.float32, device=device)

        # 2) 计算 log posterior
        log_post_prop = model.log_joint(theta_prop.unsqueeze(0))[0].item()

        # 3) MH 接受率
        log_alpha = log_post_prop - log_post
        if np.log(np.random.rand()) < log_alpha:
            # 接受
            theta = theta_prop
            log_post = log_post_prop
            accepted = 1
        else:
            accepted = 0

        accept_count += accepted
        all_samples[it, :] = theta.detach().cpu().numpy()

        # 把样本放入窗口，用于估计协方差
        window_samples.append(all_samples[it, :].copy())
        if len(window_samples) > num_cov:
            window_samples.pop(0)

        # 4) 自适应更新 scale 和 V（对应 Matlab 的 utils_update_sigma + cov）
        if it > 50:
            # 4.1 更新 scale：类似 log-scale 的 Robbins-Monro 更新
            #     log(scale_{t+1}) = log(scale_t) + γ_t (accept - target)
            gamma = 1.0 / np.sqrt(it)  # 步长递减，保证收敛
            log_scale = np.log(scale) + gamma * (accepted - target_accept)
            scale = float(np.exp(log_scale))

            # 4.2 用最近 window 的样本估计协方差
            W = np.array(window_samples)
            if W.shape[0] > 1:
                V = np.cov(W.T) + 1e-6 * np.eye(D)

        # 5) 打印进度
        if verbose and ((it + 1) % verbose == 0):
            acc_rate = accept_count / (it + 1)
            print(f"[MCMC] iter {it+1}/{n_iter}, 当前接受率 ≈ {acc_rate:.3f}, scale={scale:.4f}")

    # --- 整体接受率 ---
    accept_rate = accept_count / n_iter
    print(f"[MCMC] 最终接受率约为 {accept_rate:.3f}")

    # --- burn-in + thinning ---
    kept = all_samples[burn_in::thin, :]
    return kept
