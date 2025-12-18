from __future__ import annotations
import numpy as np
from pathlib import Path
from untils.io_utils import ensure_dir, save_csv

def generate_logistic_data(
    N: int,
    D: int,
    seed: int = 2020,
    intercept: bool = True,
    theta_scale: float = 0.1,
    standardize_x: bool = True,
    save_dir: str | Path | None = None,
):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal(size=(N, D)).astype(np.float32)

    stats = {}
    if standardize_x:
        mean = X.mean(axis=0, keepdims=True)
        std = X.std(axis=0, keepdims=True) + 1e-8
        X = (X - mean) / std
        stats = {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}

    if intercept:
        X_full = np.concatenate([np.ones((N, 1), dtype=np.float32), X], axis=1)
        theta_true = rng.normal(0.0, theta_scale, size=(D + 1,)).astype(np.float32)
    else:
        X_full = X
        theta_true = rng.normal(0.0, theta_scale, size=(D,)).astype(np.float32)

    logits = X_full @ theta_true
    p = 1.0 / (1.0 + np.exp(-logits))
    y = rng.binomial(1, p).astype(np.float32)

    if save_dir is not None:
        save_dir = ensure_dir(save_dir)
        save_csv(save_dir / "X_full.csv", X_full)
        save_csv(save_dir / "y.csv", y.reshape(-1, 1))
        save_csv(save_dir / "theta_true.csv", theta_true.reshape(-1, 1))
        if standardize_x:
            save_csv(save_dir / "x_mean.csv", stats["mean"])
            save_csv(save_dir / "x_std.csv", stats["std"])

    return X_full, y, theta_true, stats
