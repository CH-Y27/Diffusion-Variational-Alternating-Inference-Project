# generate_miss/missing_mcar.py
import numpy as np
from pathlib import Path
from untils.io_utils import save_csv, load_csv, ensure_dir

def apply_mcar_mask(
    X: np.ndarray,
    missing_rate: float,
    seed: int,
):
    """
    对 X 施加 MCAR 缺失：
        m_ij = 1 表示缺失，0 表示观测
    返回：X_miss (用 np.nan 表示缺失), mask m, X_mean_imputed
    """
    rng = np.random.default_rng(seed)
    N, D = X.shape
    mask = rng.uniform(size=(N, D)) < missing_rate  # True == 产生缺失
    X_miss = X.copy()
    X_miss[mask] = np.nan

    # 列均值填补
    col_means = np.nanmean(X_miss, axis=0)
    X_impute = X_miss.copy()
    inds = np.where(np.isnan(X_impute))
    X_impute[inds] = np.take(col_means, inds[1])

    return X_miss, mask.astype(int), X_impute

def create_missing_datasets(
    data_dir: str | Path,
    missing_rate: float = 0.3,
    seed: int = 123,
):
    data_dir = Path(data_dir)
    ensure_dir(data_dir)

    X = load_csv(data_dir / "X_full.csv")

    X_miss, m, X_impute = apply_mcar_mask(X, missing_rate, seed)

    save_csv(X_miss, data_dir / "X_miss.csv")
    save_csv(m, data_dir / "mask_m.csv")
    save_csv(X_impute, data_dir / "X_init_mean.csv")

    print(f"[Missing] 已生成 MCAR 缺失数据，缺失率={missing_rate:.2f}")
