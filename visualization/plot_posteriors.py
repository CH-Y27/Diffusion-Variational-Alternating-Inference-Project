# visualization/plot_posteriors.py
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

    可视化改进：
        - VB、MCMC 后验密度曲线下方使用浅色填充，提高美观度；
        - 每个维度的 x 轴范围设为 VB 均值 ± 4 * 标准差，
          避免展示密度极低的区域（如需更宽可自行调节倍数）。
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

    for i in range(dim_to_plot):
        ax = axes[i // 3, i % 3]

        # VB posterior 参数
        mu_vb = float(vb_mu[i])
        sigma_vb = float(vb_sigma[i])

        # 根据高斯分布特性设置 x 范围：均值 ± 4σ
        x_min = mu_vb - 4.0 * sigma_vb
        x_max = mu_vb + 4.0 * sigma_vb
        xs = np.linspace(x_min, x_max, 500)

        # VB posterior 密度
        vb_density = norm.pdf(xs, mu_vb, sigma_vb)
        # 填充 + 曲线
        ax.fill_between(xs, vb_density, alpha=0.2, color="red")
        ax.plot(xs, vb_density, color="red", label="VB")

        # MCMC posterior (kernel density estimate)
        samples = mcmc_samples[:, i]
        if len(samples) > 10:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(samples)
            kde_density = kde(xs)

            ax.fill_between(xs, kde_density, alpha=0.2, color="green")
            ax.plot(xs, kde_density, color="green", label="MCMC KDE")

        # True theta
        ax.axvline(float(theta_true[i]), color="black", linestyle="--", label="True")

        ax.set_xlim(x_min, x_max)
        ax.set_title(f"θ_{i}")
        ax.grid(True, alpha=0.3)

        if i == 0:
            ax.legend()

    # ELBO subplot
    ax_elbo = axes[2, 2]
    ax_elbo.plot(elbo_history, color="purple")
    ax_elbo.set_title("ELBO Curve")
    ax_elbo.set_xlabel("Iteration")
    ax_elbo.grid(True, alpha=0.3)

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

    可视化改进：
        - Full-VB 与 DVAI-VB 的高斯曲线均使用填充区；
        - 每个维度的 x 轴范围为两条高斯均值 ± 4σ 的并集，
          聚焦主要质量区域，便于肉眼比较。
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

    # ----------------------------------------
    # 绘制前 dim_to_plot 个参数的高斯密度
    # ----------------------------------------
    for i in range(dim_to_plot):
        ax = axes[i // 3, i % 3]

        mu_f = float(mu_full_vb[i])
        sigma_f = float(sigma_full[i])

        mu_d = float(mu_dvai[i])
        sigma_d = float(sigma_dvai[i])

        # 根据两条高斯的均值 ± 4σ 共同确定合适的显示区间
        x_min_f = mu_f - 4.0 * sigma_f
        x_max_f = mu_f + 4.0 * sigma_f
        x_min_d = mu_d - 4.0 * sigma_d
        x_max_d = mu_d + 4.0 * sigma_d

        x_min = min(x_min_f, x_min_d)
        x_max = max(x_max_f, x_max_d)
        xs = np.linspace(x_min, x_max, 500)

        # 完备数据 VB
        full_density = norm.pdf(xs, mu_f, sigma_f)
        ax.fill_between(xs, full_density, alpha=0.2, color="blue")
        ax.plot(xs, full_density, color="blue", label="Full-data VB")

        # DVAI 当前轮 VB
        dvai_density = norm.pdf(xs, mu_d, sigma_d)
        ax.fill_between(xs, dvai_density, alpha=0.2, color="red")
        ax.plot(xs, dvai_density, color="red", label="DVAI VB")

        # True θ
        ax.axvline(float(theta_true[i]), linestyle="--", color="black", label="True θ")

        if i == 0:
            ax.legend()

        ax.set_xlim(x_min, x_max)
        ax.set_title(f"θ_{i}")
        ax.grid(True, alpha=0.3)

    # ----------------------------------------
    # ELBO 曲线绘制
    # ----------------------------------------
    ax_elbo = axes[2, 2]
    ax_elbo.plot(elbo_history, color="purple")
    ax_elbo.set_title("ELBO (DVAI VB)")
    ax_elbo.set_xlabel("Iteration")
    ax_elbo.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
