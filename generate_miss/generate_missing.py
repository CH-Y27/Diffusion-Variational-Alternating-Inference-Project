# generate_missing.py
import numpy as np
from pathlib import Path
from untils.io_utils import ensure_dir, save_csv


def generate_mcar_missing(X_full, missing_rate=0.3, save_dir=None, seed=42):
    """
    生成 MCAR 缺失 + 均值填补
    参数:
        X_full: 完整数据 (N, D)
        missing_rate: 缺失比例，如 0.3
        save_dir: 保存目录，可为 None
        seed: 随机种子
    返回:
        X_obs: 含缺失值的数据 (缺失处为 np.nan)
        m_mask: 缺失掩码 (1=缺失, 0=观测)
        col_means: 每列均值
        X_imp0: 均值填补后的数据
    """
    np.random.seed(seed)
    N, D = X_full.shape

    # 掩码：1=缺失, 0=观测
    m_mask = (np.random.rand(N, D) < missing_rate).astype(np.float32)

    # 构造含缺失值矩阵
    X_obs = X_full.copy()
    X_obs[m_mask == 1] = np.nan

    # 均值填补
    col_means = np.nanmean(X_obs, axis=0)
    X_imp0 = np.where(np.isnan(X_obs), col_means, X_obs)

    # 若需要保存
    if save_dir is not None:
        save_dir = Path(save_dir)
        ensure_dir(save_dir)

        save_csv(X_obs, save_dir / "X_obs.csv")
        save_csv(m_mask, save_dir / "mask.csv")
        save_csv(X_imp0, save_dir / "X_imp0.csv")
        save_csv(col_means.reshape(1, -1), save_dir / "col_means.csv")

    return X_obs, m_mask, col_means, X_imp0
