# generate_miss/data_generation.py
from __future__ import annotations
import numpy as np
from pathlib import Path

def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -30.0, 30.0)
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float32)

def generate_logistic_data(
    N: int = None,
    D: int = None,
    seed: int = 42,
    prior_var: float = 50.0,      # 这里只是为了接口兼容；真正先验在 BayesianLogisticRegression 里用
    theta_scale: float = 1.0,
    add_intercept: bool = True,
    standardize: bool = True,
    save_dir: str | Path | None = None,

    # 兼容旧参数名：main.py 可能传 n_samples/dim_x
    n_samples: int = None,
    dim_x: int = None,
):
    # --------- 参数兼容 ----------
    if N is None:
        N = n_samples
    if D is None:
        D = dim_x
    if N is None or D is None:
        raise ValueError("Need N and D (or n_samples and dim_x).")

    rng = np.random.default_rng(seed)

    # --------- 1) 低内在维度 latent + 双簇 ----------
    d_lat = min(20, D)
    z = rng.normal(size=(N, d_lat)).astype(np.float32)
    c = rng.integers(0, 2, size=N).astype(np.float32)

    shift = np.zeros(d_lat, dtype=np.float32)
    shift[: max(2, d_lat // 5)] = 3.0
    z = z + c[:, None] * shift[None, :]

    A = rng.normal(scale=1.0 / np.sqrt(d_lat), size=(d_lat, d_lat)).astype(np.float32)
    z = z @ A

    # --------- 2) 非线性嵌入到高维 ----------
    h = 128
    W1 = rng.normal(scale=1.0 / np.sqrt(d_lat), size=(d_lat, h)).astype(np.float32)
    b1 = rng.normal(scale=0.1, size=(h,)).astype(np.float32)
    W2 = rng.normal(scale=1.0 / np.sqrt(h), size=(h, D)).astype(np.float32)
    b2 = rng.normal(scale=0.1, size=(D,)).astype(np.float32)

    X_feat = np.tanh(z @ W1 + b1) @ W2 + b2
    X_feat = X_feat + 0.10 * rng.normal(size=(N, D)).astype(np.float32)

    # --------- 3) 仅对“非截距列”标准化 ----------
    if standardize:
        mean = X_feat.mean(axis=0)
        std = X_feat.std(axis=0)
        std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
        X_feat = ((X_feat - mean) / std).astype(np.float32)

    # --------- 4) 拼接截距列：永远保持为 1，不参与标准化 ----------
    if add_intercept:
        X = np.concatenate([np.ones((N, 1), dtype=np.float32), X_feat.astype(np.float32)], axis=1)
        D_theta = D + 1
    else:
        X = X_feat.astype(np.float32)
        D_theta = D

    # --------- 5) 生成真实参数与响应 ----------
    theta_true = rng.normal(size=(D_theta,)).astype(np.float32) * float(theta_scale)
    if add_intercept:
        theta_true[0] = 0.2 * float(theta_scale)

    logits = (X @ theta_true).astype(np.float32)
    p = _sigmoid(logits)
    y = rng.binomial(1, p).astype(np.float32)

    # --------- 6) 按 main_pre.py 预期保存 CSV ----------
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # 尽量使用你项目自带的 save_csv（没有就回退到 numpy）
        try:
            from untils.io_utils import save_csv
            save_csv(save_dir / "X_full.csv", X)
            save_csv(save_dir / "y.csv", y.reshape(-1, 1))
            save_csv(save_dir / "theta_true.csv", theta_true.reshape(-1, 1))
        except Exception:
            np.savetxt(save_dir / "X_full.csv", X, delimiter=",")
            np.savetxt(save_dir / "y.csv", y.reshape(-1, 1), delimiter=",")
            np.savetxt(save_dir / "theta_true.csv", theta_true.reshape(-1, 1), delimiter=",")

    return X, y, theta_true
