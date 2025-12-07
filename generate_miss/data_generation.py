# generate_miss/data_generation.py
import numpy as np
from pathlib import Path
from untils.io_utils import ensure_dir, save_csv
from configs import data_config


def generate_linear_data(
    n_samples: int | None = None,
    dim_x: int | None = None,
    noise_std: float | None = None,
    seed: int | None = None,
    save_dir: str | Path | None = None,
    two_cluster: bool = True,        # 是否生成双峰结构
    cluster_shift: float = 20.0,     # 簇间中心大致差距（均值层面的尺度）
):
    """
    生成高维线性回归模拟数据：
        - 若 two_cluster=True，则 X 由两个高维高斯簇混合而成；
        - 两个簇使用各自不同的满秩协方差矩阵 Σ_1, Σ_2；
        - 两个簇的均值差为逐维不同的 shift_vec，而不是简单 (20,...,20) 平移。

    模型结构（two_cluster=True 时）：
        Σ_1 = A_1 A_1^T + 0.5 I_d
        Σ_2 = A_2 A_2^T + 0.5 I_d
        μ_1 ~ N(0, 10^2 I_d)
        shift_vec_j ~ N(cluster_shift, (0.3*cluster_shift)^2)
        μ_2 = μ_1 + shift_vec
        θ_true ~ N(0, I_d)
        y = X θ_true + ε,   ε ~ N(0, noise_std^2)
    """

    # -----------------------------
    # 读取默认配置
    # -----------------------------
    cfg = data_config
    n_samples = n_samples or cfg.n_samples
    dim_x     = dim_x or cfg.dim_x
    noise_std = noise_std or cfg.noise_std
    seed      = seed or cfg.seed
    save_dir  = ensure_dir(save_dir or cfg.data_dir)

    rng = np.random.default_rng(seed)

    # ============================================================
    # two_cluster = True：两个簇，各自不同协方差
    # ============================================================
    if two_cluster:
        # ---------- 1. 均值：逐维不同 ----------
        mu1 = rng.normal(loc=0.0, scale=1.0, size=dim_x)

        shift_vec = rng.normal(
            loc=cluster_shift,          # 每一维簇中心差的平均水平
            scale=cluster_shift * 0.3,  # 每一维簇中心差的随机波动
            size=dim_x
        )
        mu2 = mu1 + shift_vec

        # ---------- 2. 协方差：两个簇各自使用满秩协方差 ----------
        # 簇 1 协方差 Σ_1
        A1 = rng.normal(size=(dim_x, dim_x))
        cov1 = A1 @ A1.T + 0.5 * np.eye(dim_x)

        # 簇 2 协方差 Σ_2
        A2 = rng.normal(size=(dim_x, dim_x))
        cov2 = A2 @ A2.T + 0.5 * np.eye(dim_x)

        # ---------- 3. 样本数分配 ----------
        n1 = n_samples // 2
        n2 = n_samples - n1

        # ---------- 4. 生成两个簇 ----------
        X1 = rng.multivariate_normal(mean=mu1, cov=cov1, size=n1)
        X2 = rng.multivariate_normal(mean=mu2, cov=cov2, size=n2)

        X = np.vstack([X1, X2])

    else:
        # ========================================================
        # two_cluster = False：单一高斯簇，共用一个满秩协方差
        # ========================================================
        A = rng.normal(size=(dim_x, dim_x))
        cov_x = A @ A.T + 0.5 * np.eye(dim_x)
        X = rng.multivariate_normal(mean=np.zeros(dim_x), cov=cov_x, size=n_samples)

    # ============================================================
    # 3. 生成真实参数 θ_true ~ N(0, I_d)
    # ============================================================
    theta_true = rng.normal(loc=0.0, scale=1.0, size=dim_x)

    # ============================================================
    # 4. 生成响应变量 y = Xθ + ε
    # ============================================================
    eps = rng.normal(loc=0.0, scale=noise_std, size=n_samples)
    y = X @ theta_true + eps

    # ============================================================
    # 5. 保存结果到 CSV
    # ============================================================
    save_csv(X, save_dir / "X_full.csv")
    save_csv(y.reshape(-1, 1), save_dir / "y_full.csv")
    save_csv(theta_true.reshape(1, -1), save_dir / "theta_true.csv")

    print("[Data] 完整数据生成完成（two_cluster=True：各簇使用不同满秩协方差矩阵）。")
    print(f"[Data] 数据保存位置：{save_dir}")

    return X, y, theta_true
