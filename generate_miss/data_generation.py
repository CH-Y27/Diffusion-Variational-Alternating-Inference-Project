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
):
    """
    生成高维线性回归模拟数据：
        x ~ N(0, Σ_x) (人为构造一点相关性，让协变量“复杂一些”)
        y = x θ_true + ε,  ε ~ N(0, noise_std^2)
    并将 X, y, θ_true 保存为 CSV 方便查看
    """
    cfg = data_config
    n_samples = n_samples or cfg.n_samples
    dim_x = dim_x or cfg.dim_x
    noise_std = noise_std or cfg.noise_std
    seed = seed or cfg.seed
    save_dir = ensure_dir(save_dir or cfg.data_dir)

    rng = np.random.default_rng(seed)

    # 构造相关性的协方差矩阵 Σ_x = R R^T + 0.5 I
    A = rng.normal(size=(dim_x, dim_x))
    cov_x = A @ A.T + 0.5 * np.eye(dim_x)
    X = rng.multivariate_normal(mean=np.zeros(dim_x), cov=cov_x, size=n_samples)

    # 真实参数 θ_true
    theta_true = rng.normal(loc=0.0, scale=1.0, size=dim_x)

    # 生成 y
    eps = rng.normal(loc=0.0, scale=noise_std, size=n_samples)
    y = X @ theta_true + eps

    # 保存
    save_csv(X, save_dir / "X_full.csv")
    save_csv(y.reshape(-1, 1), save_dir / "y_full.csv")
    save_csv(theta_true.reshape(1, -1), save_dir / "theta_true.csv")

    print(f"[Data] 完整数据已生成并保存到 {save_dir}")
    return X, y, theta_true
