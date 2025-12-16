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

# VB: FullCov Gaussian
from vb.fullcov_gvb import FullCovGaussianVB

# 可视化
from visualization.plot_posteriors import plot_posterior_dvai

# 扩散填补器
from models.diffusion_imputer import SimpleDiffusionImputer


def eval_impute(X_true, X_imp, mask):
    """
    只在缺失位置上评估填补误差 (RMSE / MAE).
    X_true : 完整数据 (N, D)
    X_imp  : 当前填补数据 (N, D)
    mask   : 缺失指示 (N, D)，1=缺失，0=观测
    """
    diff = X_imp[mask == 1] - X_true[mask == 1]
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mae = float(np.mean(np.abs(diff)))
    return rmse, mae


def main():
    # ======================
    # 1. 环境 & 完整数据读取
    # ======================
    set_global_seed(data_config.seed)
    device = get_device()
    print(f"[Env] 使用设备: {device}")

    X = load_csv("data/X_full.csv")                     # (N, D)
    y = load_csv("data/y_full.csv").reshape(-1)         # (N,)
    theta_true = load_csv("data/theta_true.csv").reshape(-1)

    N, D = X.shape
    print(f"[Data] 完整数据: N={N}, D={D}")

    # ======================
    # 2. 完备数据 FullCov-GVB baseline（固定先验 N(0, I)）
    # ======================
    print("[VB-base] 完备数据 FullCov-GVB ...")
    X_t_full = torch.tensor(X, dtype=torch.float32, device=device)
    y_t_full = torch.tensor(y, dtype=torch.float32, device=device)

    model_full = BayesianLinearRegression(
        X_t_full,
        y_t_full,
        noise_var=data_config.noise_std ** 2,
        prior_var=1.0,      # 完备数据 baseline：N(0, I) 先验
    )

    vb_full = FullCovGaussianVB(dim=D, cfg=fullcov_vb_config, device=device)
    res_full = vb_full.fit(model_full)

    mu_full = res_full["mu"]        # np[D]
    cov_full = res_full["cov"]      # np[D, D]

    # ======================
    # 3. 生成缺失 + 均值填补，得到 t=0 的填补数据
    # ======================
    print("[Missing] 生成 MCAR 缺失 + 均值填补 ...")
    missing_rate = getattr(data_config, "missing_rate", 0.3)
    X_obs, m_mask, col_means, X_imp0 = generate_mcar_missing(
        X, missing_rate=missing_rate
    )
    X_imp_current = X_imp0.copy()   # t=0: 均值填补

    # ======================
    # 4. 初始化扩散填补器（加强版）
    # ======================
    imputer = SimpleDiffusionImputer(
        dim_x=D,
        hidden_dim=512,
        num_steps=50,
        J=10,
        device=device,
    )

    mu_prev = None
    cov_prev = None

    print(f"[DVAI] 外层迭代 K = {dvai_config.max_outer_iter}")
    ensure_dir("results")

    # 训练迭代数（若 dvai_config 中没有该属性，就用 150）
    diffusion_train_iters = getattr(dvai_config, "diffusion_train_iters", 150)

    # ======================
    # 5. DVAI 外层循环
    # ======================
    for t in range(dvai_config.max_outer_iter + 1):
        print(f"\n========== DVAI 迭代 t = {t} ==========")

        # 5.0 当前填补质量评估（仅在缺失位置）
        rmse_t, mae_t = eval_impute(X, X_imp_current, m_mask)
        print(f"[Impute] t={t}, RMSE={rmse_t:.6f}, MAE={mae_t:.6f}")

        # 5.1 在当前填补数据上做 VB（固定先验 N(0, I)）
        X_t = torch.tensor(X_imp_current, dtype=torch.float32, device=device)
        y_t = torch.tensor(y, dtype=torch.float32, device=device)

        model_t = BayesianLinearRegression(
            X_t,
            y_t,
            noise_var=data_config.noise_std ** 2,
            prior_var=1.0,   # 固定先验，不随 t 变化
        )

        vb_t = FullCovGaussianVB(dim=D, cfg=fullcov_vb_config, device=device)

        # warm-start：用上一轮的变分结果初始化当前参数（加速收敛）
        if mu_prev is not None and cov_prev is not None:
            vb_t.mu.data = torch.tensor(mu_prev, dtype=torch.float32, device=device)
            cov_prev_t = torch.tensor(cov_prev, dtype=torch.float32, device=device)
            jitter_ws = 1e-6 * torch.eye(D, device=device)
            L_init = torch.linalg.cholesky(cov_prev_t + jitter_ws)
            vb_t.L_unconstrained.data = L_init

        res_vb_t = vb_t.fit(model_t)
        mu_t = res_vb_t["mu"]            # np[D]
        cov_t = res_vb_t["cov"]          # np[D, D]
        elbo_t = res_vb_t["elbo_history"]

        # 5.2 可视化
        fig_path = f"results/posterior_dvai_iter{t}.png"
        plot_posterior_dvai(
            mu_full_vb=mu_full,
            cov_full_vb=cov_full,
            mu_dvai=mu_t,
            cov_dvai=cov_t,
            theta_true=theta_true,
            elbo_history=elbo_t,
            save_path=fig_path,
        )
        print(f"[Viz] t={t} 图已保存: {fig_path}")

        # 5.3 收敛检查（基于参数均值）
        if mu_prev is not None:
            delta = np.linalg.norm(mu_t - mu_prev)
            print(f"[Check] ||mu(t)-mu(t-1)|| = {delta:.6f}")
            if delta < dvai_config.convergence_tol:
                print(">>> DVAI 收敛，停止迭代")
                break

        # 保存当前轮的变分结果，供下一轮 warm-start 和最终评估
        mu_prev = mu_t.copy()
        cov_prev = cov_t.copy()

        # 最后一轮不再做扩散训练与填补
        if t == dvai_config.max_outer_iter:
            break

        # 5.4 在当前 X_imp_current 上训练扩散模型（加强版参数）
        print("[Diffusion] 训练扩散模型 ...")
        imputer.train_model(X_imp_current, iters=diffusion_train_iters)

        # 5.5 用扩散模型对当前数据进行条件填补，生成下一轮的 X_imp
        print("[Diffusion] 执行缺失填补 ...")
        X_new = imputer.impute(X_imp_current, m_mask)

        # 观测位置仍然锁死为真实观测值
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
