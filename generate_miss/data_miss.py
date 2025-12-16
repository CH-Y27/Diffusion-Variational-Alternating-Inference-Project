# data/data_miss.py
import numpy as np
from pathlib import Path
from untils.io_utils import ensure_dir, save_csv
from configs import data_config


def apply_mcar_missing(
    X_full: np.ndarray,
    missing_rate: float | None = None,
    seed: int | None = None,
):
    """
    对 X_full 施加 MCAR 缺失：
        mask_ij = 1 表示缺失，0 表示观测
    返回：
        X_miss, mask(int), X_init
    """
    cfg = data_config
    missing_rate = cfg.missing_rate if missing_rate is None else float(missing_rate)
    seed = cfg.missing_seed if seed is None else int(seed)

    rng = np.random.default_rng(seed)
    N, D = X_full.shape

    mask = (rng.uniform(size=(N, D)) < missing_rate)
    X_miss = X_full.copy()
    X_miss[mask] = np.nan

    # 初始填补（目前只做列均值）
    if cfg.impute_strategy != "col_mean":
        raise ValueError(f"Unsupported impute_strategy={cfg.impute_strategy}")

    col_means = np.nanmean(X_miss, axis=0)
    X_init = X_miss.copy()
    idx = np.where(np.isnan(X_init))
    X_init[idx] = np.take(col_means, idx[1])

    return X_miss, mask.astype(int), X_init


def save_mcar_outputs(
    X_miss: np.ndarray,
    mask: np.ndarray,
    X_init: np.ndarray,
    data_dir: str | Path | None = None,
):
    cfg = data_config
    data_dir = ensure_dir(cfg.data_dir if data_dir is None else data_dir)
    data_dir = Path(data_dir)

    save_csv(X_miss, data_dir / "X_miss.csv")
    save_csv(mask, data_dir / "mask_m.csv")
    save_csv(X_init, data_dir / "X_init_mean.csv")
