# main.py
import numpy as np
import torch
from pathlib import Path

from configs import data_config, fullcov_vb_config, dvai_config, diff_config

from untils.seed_device import set_global_seed, get_device
from untils.io_utils import ensure_dir, load_csv, save_csv

from generate_miss.data_generation import generate_logistic_data

# ====== 兼容你的 data_miss.py 不同函数名 ======
try:
    from generate_miss.data_miss import apply_mcar_missing as _make_missing
except ImportError:
    from generate_miss.data_miss import make_mcar_missing as _make_missing

try:
    from generate_miss.data_miss import save_mcar_outputs as _save_missing_outputs
except ImportError:
    _save_missing_outputs = None

from tasks.logistic_regression import BayesianLogisticRegression
from vb.fullcov_gvb import FullCovGaussianVB

from models.diffusion_imputer import DiffusionImputer, DiffusionImputerConfig

# 你的绘图函数若签名不同，请保持你本地版本调用方式
from visualization.plot_posteriors import plot_posterior_comparison


def force_intercept_col(X: np.ndarray) -> np.ndarray:
    """强制截距列=1（第0列），且任何阶段都不允许改变"""
    X[:, 0] = 1.0
    return X


def force_no_missing_on_intercept(mask_missing: np.ndarray) -> np.ndarray:
    """mask_missing: 1=missing, 0=observed；强制截距列永不缺失"""
    mask_missing[:, 0] = 0
    return mask_missing


def main():
    # ======================
    # 0) env
    # ======================
    set_global_seed(data_config.seed)
    device = get_device()
    print(f"[Env] device = {device}")

    # ======================
    # 1) dirs
    # ======================
    data_dir = Path("Data")
    res_dir = Path("Results")
    ensure_dir(data_dir)
    ensure_dir(res_dir)

    # ======================
    # 2) load / generate full data
    # ======================
    X_path = data_dir / "X_full.csv"
    y_path = data_dir / "y.csv"
    theta_path = data_dir / "theta_true.csv"

    if X_path.exists() and y_path.exists() and theta_path.exists():
        print("[Data] loading csv ...")
        X_full = load_csv(X_path).astype(np.float32)
        y = load_csv(y_path).reshape(-1).astype(np.float32)
        theta_true = load_csv(theta_path).reshape(-1).astype(np.float32)
    else:
        print("[Data] generating logistic data ...")
        X_full, y, theta_true = generate_logistic_data(
            n_samples=data_config.N,
            dim_x=data_config.D,
            seed=data_config.seed,
        )
        X_full = np.asarray(X_full, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        theta_true = np.asarray(theta_true, dtype=np.float32).reshape(-1)

        save_csv(X_path, X_full)
        save_csv(y_path, y.reshape(-1, 1))
        save_csv(theta_path, theta_true.reshape(-1, 1))

    # 强制截距列
    force_intercept_col(X_full)

    N, D = X_full.shape
    print(f"[Data] N={N}, D={D}")

    X_full_t = torch.tensor(X_full, device=device, dtype=torch.float32)
    y_t = torch.tensor(y, device=device, dtype=torch.float32)

    # ======================
    # 3) complete-data VB
    # ======================
    print("[VB] FullCov-GVB on complete data ...")
    model_full = BayesianLogisticRegression(X_full_t, y_t)
    vb_full = FullCovGaussianVB(dim=D, cfg=fullcov_vb_config, device=device)
    out_full = vb_full.fit(model_full)

    vb_mu_full = out_full["mu"]
    vb_cov_full = out_full["cov"]
    vb_elbo_full = out_full["elbo_history"]

    # ======================
    # 4) missing + mean init
    # ======================
    print("[Missing] MCAR missing + mean init ...")

    # 兼容函数名：_make_missing 可能返回 (X_miss, mask, X_init) 或别的排列
    out_miss = _make_missing(
        X_full=X_full,
        missing_rate=getattr(data_config, "missing_rate", 0.3),
        seed=getattr(data_config, "missing_seed", 123),
    )

    if len(out_miss) == 3:
        X_miss, mask_missing, X_init = out_miss
    else:
        raise RuntimeError(
            f"[Missing] make/apply missing returned {len(out_miss)} outputs, expected 3."
        )

    X_miss = np.asarray(X_miss, dtype=np.float32)
    mask_missing = np.asarray(mask_missing, dtype=np.int32)  # 1=missing,0=obs（按你模块习惯）
    X_init = np.asarray(X_init, dtype=np.float32)

    # 强制：截距列不缺失不填补
    force_no_missing_on_intercept(mask_missing)
    force_intercept_col(X_miss)
    force_intercept_col(X_init)

    # 可选保存（如果你的模块提供）
    if _save_missing_outputs is not None:
        miss_dir = data_dir / "Missing"
        ensure_dir(miss_dir)
        _save_missing_outputs(X_miss, mask_missing, X_init, data_dir=miss_dir)

    # ======================
    # 5) DVAI outer loop
    # ======================
    print(f"[DVAI] outer K={dvai_config.K}")

    # obs_mask 给 imputer：1=observed, 0=missing（与 diffusion_imputer.impute 语义对齐）
    obs_mask = 1 - mask_missing
    obs_mask_feat = obs_mask[:, 1:]  # 去掉截距列

    # 当前填补值
    X_current = X_init.copy()

    # Diffusion config（字段名按你 diffusion_imputer.py 的 DiffusionImputerConfig 对齐）
    cfg_diff = DiffusionImputerConfig(
        T=diff_config.T,
        beta_start=diff_config.beta_start,
        beta_end=diff_config.beta_end,
        hidden=diff_config.hidden,
        n_layers=diff_config.n_layers,
        t_embed_dim=diff_config.t_embed_dim,
        dropout=getattr(diff_config, "dropout", 0.0),
        lr=diff_config.lr,
        batch_size=diff_config.batch_size,
        epochs=diff_config.epochs,
        grad_clip=diff_config.grad_clip,
        ema=diff_config.ema,
        sample_steps=diff_config.sample_steps,
        num_impute_samples=diff_config.num_impute_samples,
        clamp_val=diff_config.clamp_val,
    )

    # 只对非截距列做扩散
    imputer = DiffusionImputer(x_dim=D - 1, device=device, cfg=cfg_diff)

    mu_prev = vb_mu_full.copy()

    for k in range(dvai_config.K):
        print(f"\n========== DVAI iter {k} ==========")

        # (a) VB on current imputed data
        Xk_t = torch.tensor(X_current, device=device, dtype=torch.float32)
        model_k = BayesianLogisticRegression(Xk_t, y_t)

        vb_k = FullCovGaussianVB(dim=D, cfg=fullcov_vb_config, device=device)
        out_k = vb_k.fit(model_k)
        mu_k = out_k["mu"]

        diff_mu = float(np.linalg.norm(mu_k - mu_prev))
        print(f"[Check] ||mu_k - mu_prev|| = {diff_mu:.6f}")
        mu_prev = mu_k

        # (b) diffusion train (fit) on features only
        print("[Diffusion] training ...")
        X_train_feat = X_current[:, 1:].astype(np.float32)
        imputer.fit(X_train_feat, verbose=True)  # ✅ 你的类没有 train，只有 fit

        # (c) diffusion impute features (inpainting)
        print("[Diffusion] imputing ...")
        X_obs_feat = X_current[:, 1:].astype(np.float32)
        X_imp_feat = imputer.impute(X_obs=X_obs_feat, obs_mask=obs_mask_feat)

        # (d) damping，防止每轮填补导致后验跑飞
        eta = getattr(dvai_config, "impute_damping", 0.2)
        X_current[:, 1:] = (1 - eta) * X_current[:, 1:] + eta * X_imp_feat
        force_intercept_col(X_current)

    # ======================
    # 6) plot (complete-data reference)
    # ======================
    fig_path = res_dir / "posterior_compare_complete_logistic.png"
    plot_posterior_comparison(
        theta_true=theta_true,
        vb_mu=vb_mu_full,
        vb_cov=vb_cov_full,
        mcmc_samples=None,
        elbo_history=vb_elbo_full,
        save_path=str(fig_path),
        max_dim=8,
    )
    print(f"[Plot] saved to: {fig_path}")


if __name__ == "__main__":
    main()
