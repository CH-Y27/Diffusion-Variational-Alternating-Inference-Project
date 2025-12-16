# generate_miss/data_generation.py
import numpy as np
from pathlib import Path
from untils.io_utils import ensure_dir, save_csv
from configs import data_config


def generate_logistic_data(
    n_samples: int | None = None,
    dim_x: int | None = None,
    seed: int | None = None,
    save_dir: str | Path | None = None,
):
    """
    生成 logistic 回归模拟数据（含截距）并保存：
        X_full.csv, y_full.csv, theta_true.csv
    """
    cfg = data_config

    n_samples = cfg.n_samples if n_samples is None else n_samples
    dim_x     = cfg.dim_x if dim_x is None else dim_x
    seed      = cfg.seed if seed is None else seed
    save_dir  = ensure_dir(cfg.data_dir if save_dir is None else save_dir)

    rng = np.random.default_rng(seed)

    # ============================================================
    # 1) 生成协变量 X
    # ============================================================
    if cfg.manifold:
        latent_dim = cfg.latent_dim

        if cfg.two_cluster:
            mu1_z = rng.normal(0.0, 1.0, size=latent_dim)
            shift_vec_z = rng.normal(cfg.cluster_shift, 0.3 * cfg.cluster_shift, size=latent_dim)
            mu2_z = mu1_z + shift_vec_z

            A1z = rng.normal(size=(latent_dim, latent_dim))
            cov1_z = A1z @ A1z.T + 0.5 * np.eye(latent_dim)

            A2z = rng.normal(size=(latent_dim, latent_dim))
            cov2_z = A2z @ A2z.T + 0.5 * np.eye(latent_dim)

            n1 = n_samples // 2
            n2 = n_samples - n1
            Z1 = rng.multivariate_normal(mu1_z, cov1_z, size=n1)
            Z2 = rng.multivariate_normal(mu2_z, cov2_z, size=n2)
            Z = np.vstack([Z1, Z2])
        else:
            Azz = rng.normal(size=(latent_dim, latent_dim))
            cov_z = Azz @ Azz.T + 0.5 * np.eye(latent_dim)
            Z = rng.multivariate_normal(np.zeros(latent_dim), cov_z, size=n_samples)

        # 非线性嵌入
        W1 = rng.normal(0.0, 1.0 / np.sqrt(latent_dim), size=(latent_dim, cfg.embed_hidden))
        b1 = rng.normal(0.0, 0.1, size=(cfg.embed_hidden,))
        W2 = rng.normal(0.0, 1.0 / np.sqrt(cfg.embed_hidden), size=(cfg.embed_hidden, dim_x))
        b2 = rng.normal(0.0, 0.1, size=(dim_x,))

        H = np.tanh(Z @ W1 + b1)
        X = H @ W2 + b2

        # 标准化 + 流形厚度噪声
        X = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-8)
        X = X + rng.normal(0.0, cfg.manifold_noise, size=X.shape)

    else:
        A = rng.normal(size=(dim_x, dim_x))
        cov_x = A @ A.T + 0.5 * np.eye(dim_x)
        X = rng.multivariate_normal(np.zeros(dim_x), cov_x, size=n_samples)

    # ============================================================
    # 2) 生成 theta_true（含截距）
    # ============================================================
    theta0 = rng.normal(0.0, cfg.theta0_std)
    theta  = rng.normal(0.0, cfg.theta_std, size=dim_x)
    theta_true = np.concatenate([[theta0], theta], axis=0)

    # ============================================================
    # 3) 生成 y（Bernoulli-logistic）
    # ============================================================
    clip = float(cfg.sigmoid_clip)
    logits = theta0 + X @ theta
    p = 1.0 / (1.0 + np.exp(-np.clip(logits, -clip, clip)))
    y = rng.binomial(1, p, size=n_samples).astype(float)

    # ============================================================
    # 4) 保存
    # ============================================================
    save_csv(X, Path(save_dir) / "X_full.csv")
    save_csv(y.reshape(-1, 1), Path(save_dir) / "y_full.csv")
    save_csv(theta_true.reshape(1, -1), Path(save_dir) / "theta_true.csv")

    print(f"[Data] Logistic data generated: N={n_samples}, D={dim_x}, d_latent={cfg.latent_dim}, manifold={cfg.manifold}")
    print(f"[Data] Saved to: {save_dir}")

    return X, y, theta_true
