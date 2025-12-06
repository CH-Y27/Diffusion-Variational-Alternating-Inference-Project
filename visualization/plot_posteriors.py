import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


# ============================================================
# 1. 基线实验绘图 (FullCov-VB vs MCMC vs True)
#    main_pre.py 使用
# ============================================================

def plot_posterior_comparison(
    theta_true,
    vb_mu,
    vb_cov,
    mcmc_samples,
    elbo_history,
    save_path,
    max_dim=8
):
    """
    绘制：
        - True 参数
        - VB 后验 N(mu, cov)
        - MCMC 样本密度
        - ELBO 曲线
    """

    # ------ 统一形状为 1D / diag ------
    theta_true = np.asarray(theta_true).reshape(-1)
    vb_mu = np.asarray(vb_mu).reshape(-1)

    vb_cov_arr = np.asarray(vb_cov)
    if vb_cov_arr.ndim == 2:
        vb_var = np.diag(vb_cov_arr)
    else:  # 1D 当作方差向量
        vb_var = vb_cov_arr.reshape(-1)
    vb_sigma = np.sqrt(np.clip(vb_var, 1e-12, None))

    mcmc_samples = np.asarray(mcmc_samples)

    D = min(len(theta_true), len(vb_mu), mcmc_samples.shape[1])
    dim_to_plot = min(max_dim, D)

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    xs = np.linspace(-3, 3, 500)

    for i in range(dim_to_plot):
        ax = axes[i // 3, i % 3]

        # VB posterior
        mu_vb = float(vb_mu[i])
        sigma_vb = float(vb_sigma[i])
        ax.plot(xs, norm.pdf(xs, mu_vb, sigma_vb),
                color="red", label="VB")

        # MCMC posterior (kernel density estimate)
        samples = mcmc_samples[:, i]
        if len(samples) > 10:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(samples)
            ax.plot(xs, kde(xs), color="green", label="MCMC KDE")

        # True theta
        ax.axvline(float(theta_true[i]), color="black", linestyle="--", label="True")

        ax.set_title(f"θ_{i}")
        ax.grid(True)

        if i == 0:
            ax.legend()

    # ELBO subplot
    ax_elbo = axes[2, 2]
    ax_elbo.plot(elbo_history, color="purple")
    ax_elbo.set_title("ELBO Curve")
    ax_elbo.set_xlabel("Iteration")
    ax_elbo.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()



# ============================================================
# 2. DVAI 实验绘图 (Full VB vs DVAI VB vs True)
#    main.py 使用
# ============================================================

def plot_posterior_dvai(
    mu_full_vb,
    cov_full_vb,
    mu_dvai,
    cov_dvai,
    theta_true,
    elbo_history,
    save_path,
    max_dim=8
):
    """
    画出 DVAI 每轮 VB 后验 与 完备 VB 后验、真实参数的对比。
    风格与 main_pre.py 一致：
        - 蓝色：Full-data VB 高斯密度
        - 红色：DVAI 当前轮 VB 高斯密度
        - 黑色虚线：真实 θ
        - 右下角：本轮 VB 的 ELBO 曲线
    """

    # ---------- 统一为 1D 向量 ----------
    theta_true = np.asarray(theta_true).reshape(-1)
    mu_full_vb = np.asarray(mu_full_vb).reshape(-1)
    mu_dvai = np.asarray(mu_dvai).reshape(-1)

    # 协方差：支持 full matrix 或 diag vector
    def extract_sigma(cov, D_target):
        cov_arr = np.asarray(cov)
        if cov_arr.ndim == 2:
            var = np.diag(cov_arr)
        else:
            var = cov_arr.reshape(-1)
        var = var[:D_target]
        return np.sqrt(np.clip(var, 1e-12, None))

    D = min(len(theta_true), len(mu_full_vb), len(mu_dvai))
    dim_to_plot = min(max_dim, D)

    sigma_full = extract_sigma(cov_full_vb, D)
    sigma_dvai = extract_sigma(cov_dvai, D)

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    xs = np.linspace(-3, 3, 500)

    # ----------------------------------------
    # 绘制前 dim_to_plot 个参数的高斯密度
    # ----------------------------------------
    for i in range(dim_to_plot):
        ax = axes[i // 3, i % 3]

        mu_f = float(mu_full_vb[i])
        sigma_f = float(sigma_full[i])

        mu_d = float(mu_dvai[i])
        sigma_d = float(sigma_dvai[i])

        # 完备数据 VB
        ax.plot(xs, norm.pdf(xs, mu_f, sigma_f),
                color="blue", label="Full-data VB")

        # DVAI 当前轮 VB
        ax.plot(xs, norm.pdf(xs, mu_d, sigma_d),
                color="red", label="DVAI VB")

        # True θ
        ax.axvline(float(theta_true[i]), linestyle="--", color="black", label="True θ")

        if i == 0:
            ax.legend()

        ax.set_title(f"θ_{i}")
        ax.grid(True)

    # ----------------------------------------
    # ELBO 曲线绘制
    # ----------------------------------------
    ax_elbo = axes[2, 2]
    ax_elbo.plot(elbo_history, color="purple")
    ax_elbo.set_title("ELBO (DVAI VB)")
    ax_elbo.set_xlabel("Iteration")
    ax_elbo.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
