# main.py
import numpy as np
import torch

# configs
from configs import data_config, fullcov_vb_config, dvai_config

# utils
from untils.seed_device import set_global_seed, get_device
from untils.io_utils import load_csv, ensure_dir

# missing data
from generate_miss.generate_missing import generate_mcar_missing

# 线性回归贝叶斯模型（带 log_joint 方法）
from tasks.linear_regression import BayesianLinearRegression

# VB: FullCov Gaussian VB
from vb.fullcov_gvb import FullCovGaussianVB

# 可视化（使用你原来的版本，不改）
from visualization.plot_posteriors import plot_posterior_dvai

# 扩散填补器
from models.diffusion_imputer import SimpleDiffusionImputer


def main():
    # ======================
    # 1. 环境 & 数据读取
    # ======================
    set_global_seed(42)
    device = get_device()
    print(f"[Env] 使用设备: {device}")

    X = load_csv("data/X_full.csv")
    y = load_csv("data/y_full.csv").reshape(-1)
    theta_true = load_csv("data/theta_true.csv").reshape(-1)

    N, D = X.shape
    print(f"[Data] 完整数据: N={N}, D={D}")

    # ======================
    # 2. 完备数据 FullCov-GVB baseline
    # ======================
    print("[VB-base] 完备数据 FullCov-GVB ...")
    X_t_full = torch.tensor(X, dtype=torch.float32, device=device)
    y_t_full = torch.tensor(y, dtype=torch.float32, device=device)

    model_full = BayesianLinearRegression(
        X_t_full,
        y_t_full,
        noise_var=data_config.noise_std ** 2,
        prior_var=1.0,
    )

    vb_full = FullCovGaussianVB(dim=D, cfg=fullcov_vb_config, device=device)
    res_full = vb_full.fit(model_full)

    mu_full = res_full["mu"]
    cov_full = res_full["cov"]

    # ======================
    # 3. 生成缺失 + 均值填补，得到 t=0 的 X_imp
    # ======================
    print("[Missing] 生成 MCAR 缺失 + 均值填补 ...")
    X_obs, m_mask, col_means, X_imp0 = generate_mcar_missing(X, missing_rate=0.3)
    X_imp_current = X_imp0.copy()   # t=0 的填补数据

    # ======================
    # 4. 初始化扩散填补器（后续 warm-start）
    # ======================
    imputer = SimpleDiffusionImputer(
        dim_x=D,
        hidden_dim=256,
        num_steps=30,
        J=3,
        device=device,
    )

    mu_prev = None
    print(f"[DVAI] 外层迭代 K = {dvai_config.max_outer_iter}")
    ensure_dir("results")

    # ======================
    # 5. DVAI 外层循环
    # ======================
    for t in range(dvai_config.max_outer_iter + 1):
        print(f"\n========== DVAI 迭代 t = {t} ==========")

        # 5.1 当前填补数据上做 VB
        X_t = torch.tensor(X_imp_current, dtype=torch.float32, device=device)
        y_t = torch.tensor(y, dtype=torch.float32, device=device)

        model_t = BayesianLinearRegression(
            X_t,
            y_t,
            noise_var=data_config.noise_std ** 2,
            prior_var=1.0,
        )

        vb_t = FullCovGaussianVB(dim=D, cfg=fullcov_vb_config, device=device)
        res_vb_t = vb_t.fit(model_t)

        mu_t = res_vb_t["mu"]
        cov_t = res_vb_t["cov"]
        elbo_t = res_vb_t["elbo_history"]

        # 5.2 可视化（沿用你原来的函数签名）
        fig_path = f"results/posterior_dvai_iter{t}.png"
        plot_posterior_dvai(
            theta_true,
            mu_full,
            cov_full,
            mu_t,
            cov_t,
            elbo_t,
            fig_path,
        )
        print(f"[Viz] t={t} 图已保存: {fig_path}")

        # 5.3 收敛检查
        if mu_prev is not None:
            delta = np.linalg.norm(mu_t - mu_prev)
            print(f"[Check] ||mu(t)-mu(t-1)|| = {delta:.6f}")
            if delta < dvai_config.convergence_tol:
                print(">>> DVAI 收敛，停止迭代")
                break

        mu_prev = mu_t.copy()

        # 最后一轮不再训练扩散 & 填补
        if t == dvai_config.max_outer_iter:
            break

        # 5.4 扩散模型训练：在当前 X_imp_current 上（原尺度，内部会标准化）
        print("[Diffusion] 训练扩散模型 ...")
        imputer.train_model(X_imp_current, iters=80)

        # 5.5 扩散填补：基于上一轮的 X_imp_current 进行 refinement
        print("[Diffusion] 执行缺失填补 ...")
        X_new = imputer.impute(X_imp_current, m_mask)

        # 观测位置永远锁定为真实观测值
        X_new[m_mask == 0] = X_obs[m_mask == 0]

        X_imp_current = X_new.copy()

    # ======================
    # 6. 最终评估
    # ======================
    print("\n========== DVAI 最终评估 ==========")
    print("|| 完备数据 VB - 真值 || =",
          np.linalg.norm(mu_full - theta_true))
    print("|| DVAI 最终参数 - 真值 || =",
          np.linalg.norm(mu_prev - theta_true))
    print("|| DVAI 最终参数 - 完备 VB || =",
          np.linalg.norm(mu_prev - mu_full))
    print("[Done] DVAI 全部完成！")


if __name__ == "__main__":
    main()
