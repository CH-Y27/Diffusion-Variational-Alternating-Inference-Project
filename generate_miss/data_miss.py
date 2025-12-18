# generate_miss/data_miss.py
import numpy as np
from pathlib import Path

from untils.io_utils import ensure_dir, save_csv


def make_mcar_missing(
    X_full: np.ndarray,
    missing_rate: float,
    seed: int = 42,
    skip_first_col: bool = True,
):
    """
    MCAR 缺失：对每个元素独立以 missing_rate 概率置为 nan。
    注意：若 skip_first_col=True，则第0列（截距列，全1）不产生缺失。
    Returns:
        X_miss: 含 nan 的数组
        M: mask, True 表示“缺失位置”，False 表示“观测位置”
    """
    rng = np.random.default_rng(seed)
    X_full = np.asarray(X_full, dtype=np.float32)

    M = rng.random(X_full.shape) < float(missing_rate)

    if skip_first_col:
        M[:, 0] = False  # 截距列永不缺失

    X_miss = X_full.copy()
    X_miss[M] = np.nan

    # 截距列强制保持原样（一般是全1）
    if skip_first_col:
        X_miss[:, 0] = X_full[:, 0]

    return X_miss, M


def mean_impute(
    X_miss: np.ndarray,
    M: np.ndarray,
    skip_first_col: bool = True,
):
    """
    列均值填补：对每列用“非缺失值”的均值填补缺失项。
    注意：第0列（截距列）不参与填补，强制保持原值（通常全1）。
    """
    X_miss = np.asarray(X_miss, dtype=np.float32)
    X_imp = X_miss.copy()

    # 逐列填补
    D = X_imp.shape[1]
    for j in range(D):
        if skip_first_col and j == 0:
            continue

        col = X_imp[:, j]
        obs = ~np.isnan(col)
        if obs.sum() == 0:
            # 极端情况：整列全缺失 -> 用0兜底（也可改成别的策略）
            mu = 0.0
        else:
            mu = float(col[obs].mean())

        miss_idx = np.isnan(col)
        col[miss_idx] = mu
        X_imp[:, j] = col

    if skip_first_col:
        # 截距列保持不变（避免任何数值污染）
        X_imp[:, 0] = X_miss[:, 0]

    return X_imp


def build_missing_and_init_impute(
    X_full: np.ndarray,
    missing_rate: float,
    seed: int,
    out_dir: Path,
    skip_first_col: bool = True,
):
    """
    生成缺失 + 均值填补，并保存：
      - X_miss.csv  (含nan)
      - M.csv       (0/1, 1表示缺失)
      - X_init.csv  (初始填补，用于DVAI第一轮)
    """
    out_dir = Path(out_dir)
    ensure_dir(out_dir)

    X_miss, M = make_mcar_missing(
        X_full=X_full,
        missing_rate=missing_rate,
        seed=seed,
        skip_first_col=skip_first_col,
    )
    X_init = mean_impute(X_miss, M, skip_first_col=skip_first_col)

    # 保存：注意 save_csv(path, array)
    save_csv(out_dir / "X_miss.csv", X_miss)
    save_csv(out_dir / "M.csv", M.astype(np.int32))
    save_csv(out_dir / "X_init.csv", X_init)

    return X_miss, M, X_init
