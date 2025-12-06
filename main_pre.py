# main_pre.py
import numpy as np
import torch
from pathlib import Path

# === configs ===
from configs import (
    data_config,
    fullcov_vb_config,
    mcmc_config,
)

# === utils ===
from untils.seed_device import set_global_seed, get_device
from untils.io_utils import ensure_dir, load_csv

# === data generation ===
from generate_miss.data_generation import generate_linear_data

# === model ===
from tasks.linear_regression import BayesianLinearRegression

# === VB: FullCov Gaussian ===
from vb.fullcov_gvb import FullCovGaussianVB

# === MCMC ===
from vb.mcmc_linear import run_mcmc_linear

# === Analytic posterior ===
from vb.analytic_posterior import analytic_posterior_linear

# === Visualization ===
from visualization.plot_posteriors import plot_posterior_comparison


def main():

    # ===============================================
    # 1. 环境设置
    # ===============================================
    set_global_seed(data_config.seed)
    device = get_device()
    print(f"[Env] 使用设备: {device}")

    data_dir = Path(data_config.data_dir)
    ensure_dir(data_dir)

    # ===============================================
    # 2. 加载或生成完备数据
    # ===============================================
    X_path = data_dir / "X_full.csv"
    y_path = data_dir / "y_full.csv"
    theta_path = data_dir / "theta_true.csv"

    if not X_path.exists():
        print("[Data] 未找到完整数据，正在生成 ...")
        X, y, theta_true = generate_linear_data()
    else:
        print("[Data] 读取已有完整数据 ...")
        X = load_csv(X_path)
        y = load_csv(y_path).reshape(-1)
        theta_true = load_csv(theta_path).reshape(-1)

    N, D = X.shape
    print(f"[Data] 完整数据: N={N}, D={D}")

    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    y_t = torch.tensor(y, dtype=torch.float32, device=device)

    # ===============================================
    # 3. 解析后验（真实后验，用于检验 VB）
    # ===============================================
    mu_post, Sigma_post = analytic_posterior_linear(
        X, y,
        noise_var=data_config.noise_std ** 2,
        prior_var=1.0,
    )

    # ===============================================
    # 4. 构建线性模型（供 VB 和 MCMC 使用）
    # ===============================================
    model = BayesianLinearRegression(
        X_t, y_t,
        noise_var=data_config.noise_std ** 2,
        prior_var=1.0,
    )

    # ===============================================
    # 5. FullCov-GVB 推断
    # ===============================================
    print("[VB] 使用 FullCov-GVB 进行高斯变分推断 ...")
    vb = FullCovGaussianVB(dim=D, cfg=fullcov_vb_config, device=device)
    vb_result = vb.fit(model)

    vb_mu = vb_result["mu"]
    vb_cov = vb_result["cov"]
    elbo_history = vb_result["elbo_history"]

    # ===============================================
    # 6. 运行 MCMC（随机游走 Metropolis）
    # ===============================================
    print("[MCMC] 开始采样 ...")
    mcmc_samples = run_mcmc_linear(
        model, cfg=mcmc_config, seed=data_config.seed + 1
    )
    mcmc_mean = mcmc_samples.mean(axis=0)

    # ===============================================
    # 7. 数值评估
    # ===============================================
    print("\n========== 数值评估 ==========")
    print("[Eval] ||解析后验 mean - true θ||_2 =",
          np.linalg.norm(mu_post - theta_true))
    print("[Eval] ||VB (FullCov) mean - 解析后验 mean||_2 =",
          np.linalg.norm(vb_mu - mu_post))
    print("[Eval] ||MCMC mean - 解析后验 mean||_2 =",
          np.linalg.norm(mcmc_mean - mu_post))

    # ===============================================
    # 8. 可视化（前 8 个参数 + ELBO）
    # ===============================================
    ensure_dir("results")
    fig_path = "results/posterior_complete_3x3_fullcov.png"

    plot_posterior_comparison(
        theta_true=theta_true,
        vb_mu=vb_mu,
        vb_cov=vb_cov,
        mcmc_samples=mcmc_samples,
        elbo_history=elbo_history,
        save_path=fig_path,
    )

    print(f"\n[Viz] 后验比较图已保存到: {fig_path}")
    print("[Done] FullCov-GVB 完备数据检验已完成。\n")


if __name__ == "__main__":
    main()
