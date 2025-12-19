# generate_miss/data_miss.py
"""
缺失机制与初始填补（本项目优先 MCAR）。
约定：
- mask: bool, True 表示该位置“缺失/需要填补”
- 截距列（col=0）永不缺失、永不填补
"""
from __future__ import annotations
import numpy as np
from pathlib import Path
from typing import Tuple


def make_mcar_missing(
    X: np.ndarray,
    missing_rate: float,
    seed: int = 42,
    skip_col0: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    mask = rng.random(size=X.shape) < float(missing_rate)
    if skip_col0 and X.shape[1] > 0:
        mask[:, 0] = False
    X_obs = X.astype(np.float32).copy()
    X_obs[mask] = np.nan
    return X_obs, mask


# 兼容旧名字：之前你 main.py 里 import apply_mcar_missing
def apply_mcar_missing(*args, **kwargs):
    return make_mcar_missing(*args, **kwargs)


def mean_impute(
    X_obs: np.ndarray,
    mask: np.ndarray,
    skip_col0: bool = True,
) -> np.ndarray:
    X_imp = X_obs.copy().astype(np.float32)
    D = X_imp.shape[1]
    start = 1 if skip_col0 else 0

    for j in range(start, D):
        col = X_imp[:, j]
        obs = ~mask[:, j]
        fill = 0.0 if obs.sum() == 0 else np.nanmean(col[obs]).astype(np.float32)
        miss = mask[:, j]
        col[miss] = fill
        X_imp[:, j] = col

    if skip_col0 and D > 0:
        X_imp[:, 0] = 1.0
    return X_imp


def save_mcar_outputs(
    out_dir: str | Path,
    X_full: np.ndarray,
    y: np.ndarray,
    X_obs: np.ndarray,
    mask: np.ndarray,
    X_init: np.ndarray,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "X_full.npy", X_full)
    np.save(out_dir / "y.npy", y)
    np.save(out_dir / "X_obs.npy", X_obs)
    np.save(out_dir / "mask.npy", mask.astype(np.bool_))
    np.save(out_dir / "X_init.npy", X_init)
