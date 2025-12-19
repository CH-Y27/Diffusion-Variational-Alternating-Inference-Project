# main.py
from __future__ import annotations

import csv
from pathlib import Path
import numpy as np
import torch

from configs import data_config, fullcov_vb_config, dvai_config, diff_config
from untils.seed_device import set_global_seed, get_device
from untils.io_utils import ensure_dir, load_csv, save_csv

from generate_miss.data_generation import generate_logistic_data
from generate_miss.data_miss import make_mcar_missing, mean_impute, save_mcar_outputs

from tasks.logistic_regression import BayesianLogisticRegression
from vb.fullcov_gvb import FullCovGaussianVB, FullCovGVBConfig

from models.diffusion_imputer import DiffusionImputer, DiffusionImputerConfig
from visualization.plot_dvai import plot_posterior_dvai_comparison


def rmse(a, b):
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    return float(np.sqrt(np.mean((a - b) ** 2)))

def mae(a, b):
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    return float(np.mean(np.abs(a - b)))

def mse(a, b):
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    return float(np.mean((a - b) ** 2))

def force_intercept(X: np.ndarray):
    X[:, 0] = 1.0
    return X


def _map_init_lbfgs(model: BayesianLogisticRegression, iters: int = 200):
    device = model.X.device
    theta = torch.zeros(model.D, device=device, requires_grad=True)
    opt = torch.optim.LBFGS([theta], lr=1.0, max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad(set_to_none=True)
        loss = -model.log_joint(theta)[0]
        loss.backward()
        return loss

    opt.step(closure)
    return theta.detach()


def compute_mean_std_from_observed(X_obs_feat: np.ndarray, mask_feat: np.ndarray):
    """
    X_obs_feat: (N,D) with nan at missing
    mask_feat:  (N,D) bool, True=missing
    """
    mean = np.nanmean(X_obs_feat, axis=0).astype(np.float32)
    std = np.nanstd(X_obs_feat, axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return mean, std


def save_param_table(path: Path, mu, cov, mu_full, cov_full):
    var = np.diag(cov).copy()
    var_full = np.diag(cov_full).copy()
    D = len(mu)

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dim", "mu", "var", "mu_full", "var_full", "delta_mu", "delta_var"])
        for i in range(D):
            w.writerow([i, float(mu[i]), float(var[i]), float(mu_full[i]), float(var_full[i]),
                        float(mu[i] - mu_full[i]), float(var[i] - var_full[i])])


def kl_diag_gaussians(mu0, v0, mu1, v1):
    """
    KL(N(mu0,diag(v0)) || N(mu1,diag(v1)))
    """
    mu0 = np.asarray(mu0).reshape(-1)
    mu1 = np.asarray(mu1).reshape(-1)
    v0 = np.asarray(v0).reshape(-1)
    v1 = np.asarray(v1).reshape(-1)
    v0 = np.maximum(v0, 1e-12)
    v1 = np.maximum(v1, 1e-12)

    term = np.log(v1 / v0) + (v0 + (mu0 - mu1) ** 2) / v1 - 1.0
    return float(0.5 * term.sum())


def main():
    # ======================
    # 0) env + dirs
    # ======================
    set_global_seed(data_config.seed)
    device = get_device()
    print(f"[Env] device = {device}")

    data_dir = Path("Data")
    res_dir = Path("Results")
    dvai_dir = data_dir / "DVAI"
    ensure_dir(data_dir)
    ensure_dir(res_dir)
    ensure_dir(dvai_dir)

    # ======================
    # 1) load / generate full data
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
            N=data_config.N,
            D=data_config.D,
            seed=data_config.seed,
            prior_var=data_config.prior_var,
            theta_scale=data_config.theta_scale,
            add_intercept=data_config.add_intercept,
            standardize=data_config.standardize,
            save_dir=data_dir,
        )
        save_csv(X_path, X_full)
        save_csv(y_path, y.reshape(-1, 1))
        save_csv(theta_path, theta_true.reshape(-1, 1))

    force_intercept(X_full)
    N, D = X_full.shape
    print(f"[Data] N={N}, D={D}")

    X_full_t = torch.tensor(X_full, device=device, dtype=torch.float32)
    y_t = torch.tensor(y, device=device, dtype=torch.float32)

    # ======================
    # 2) baseline: MAP + VB on complete (标准)
    # ======================
    print("[Baseline] MAP init (complete) ...")
    model_full_map = BayesianLogisticRegression(X_full_t, y_t, prior_var=data_config.prior_var)
    theta_map_full = _map_init_lbfgs(model_full_map, iters=200)

    print("[Baseline] VB on complete data ...")
    vb_cfg_full = FullCovGVBConfig(
        lr=fullcov_vb_config.lr,
        max_iter=fullcov_vb_config.max_iter,
        num_mc=fullcov_vb_config.num_mc,
        grad_clip=fullcov_vb_config.grad_clip,
        window=fullcov_vb_config.window,
        seed=fullcov_vb_config.seed,
        verbose=fullcov_vb_config.verbose,
    )
    vb_full = FullCovGaussianVB(dim=D, cfg=vb_cfg_full, device=device)
    with torch.no_grad():
        vb_full.mu.copy_(theta_map_full)
    out_full = vb_full.fit(model_full_map)

    mu_full = out_full["mu"]
    cov_full = out_full["cov"]

    # ======================
    # 3) missing (MCAR) + DiffPuter-style mean/std + init fill=0 (standardized)
    # ======================
    print("[Missing] MCAR + init ...")
    X_obs, mask = make_mcar_missing(
        X_full, missing_rate=float(data_config.missing_rate),
        seed=int(data_config.missing_seed), skip_col0=True
    )
    # mean-impute只是用于“VB第0轮对比”，但扩散条件输入我们用“标准化后缺失置0”
    X_init_mean = mean_impute(X_obs, mask, skip_col0=True)

    save_mcar_outputs(dvai_dir, X_full, y, X_obs, mask, X_init_mean)

    mask_feat = mask[:, 1:]                       # True=missing
    obs_mask_feat = (~mask_feat).astype(np.float32)  # 1 observed, 0 missing

    X_obs_feat = X_obs[:, 1:].astype(np.float32)  # with nan
    mean_feat, std_feat = compute_mean_std_from_observed(X_obs_feat, mask_feat)

    # ------------------------------------------------------------------
    # EDM / score-model scale alignment (DiffPuter-style)
    #
    # EDM uses sigma_data≈0.5 by default. If we only standardize to unit-variance,
    # the typical magnitude is not well aligned with sigma_data, which makes
    # training & sampling noticeably less stable (especially on CPU / small epochs).
    # A simple and robust fix is an extra division by 2.
    #
    # IMPORTANT: This scaling is ONLY for diffusion internal space.
    # We always convert back to the original feature space before feeding VB.
    # ------------------------------------------------------------------
    EDM_SCALE = 2.0

    def to_std(x_feat: np.ndarray) -> np.ndarray:
        x = (x_feat - mean_feat) / std_feat
        return (x / EDM_SCALE).astype(np.float32)

    def from_std(x_std: np.ndarray) -> np.ndarray:
        x = x_std * EDM_SCALE
        return (x * std_feat + mean_feat).astype(np.float32)

    # X_current：用于VB的“原尺度”
    X_current = X_init_mean.copy().astype(np.float32)
    force_intercept(X_current)

    # ======================
    # 4) VB iter0: mean-imputed (先验与完备数据一致 + 初始化一致)
    # ======================
    print("[Init-VB] VB on mean-imputed (same prior & init as complete) ...")
    model_0 = BayesianLogisticRegression(
        torch.tensor(X_current, device=device, dtype=torch.float32),
        y_t,
        prior_var=data_config.prior_var
    )

    vb_cfg0 = vb_cfg_full
    vb_0 = FullCovGaussianVB(dim=D, cfg=vb_cfg0, device=device)
    with torch.no_grad():
        vb_0.mu.copy_(theta_map_full)   # ✅ 与完备数据VB初始化一致
    out_0 = vb_0.fit(model_0)
    mu_prev, cov_prev = out_0["mu"], out_0["cov"]
    elbo_prev = out_0["elbo_history"]

    save_param_table(res_dir / "params_iter_0.csv", mu_prev, cov_prev, mu_full, cov_full)
    plot_posterior_dvai_comparison(
        theta_true=theta_true,
        mu_full=mu_full, cov_full=cov_full,
        mu_dvai=mu_prev, cov_dvai=cov_prev,
        elbo_history=elbo_prev,
        save_path=str(res_dir / "dvai_iter_0_posterior.png"),
        max_dim=8,
    )

    # ======================
    # 5) build diffusion imputer
    # ======================
    cfg_diff = DiffusionImputerConfig(
        T=diff_config.T,
        beta_start=diff_config.beta_start,
        beta_end=diff_config.beta_end,
        hidden=diff_config.hidden,
        n_layers=diff_config.n_layers,
        t_embed_dim=diff_config.t_embed_dim,
        dropout=diff_config.dropout,
        lr=diff_config.lr,
        batch_size=diff_config.batch_size,
        epochs=diff_config.epochs,
        grad_clip=diff_config.grad_clip,
        ema=diff_config.ema,
        sample_steps=diff_config.sample_steps,
        num_impute_samples=diff_config.num_impute_samples,
        clamp_val=diff_config.clamp_val,
    )
    imputer = DiffusionImputer(x_dim=D - 1, device=device, cfg=cfg_diff)

    # ======================
    # 6) DVAI loop (k=1..K)
    # ======================
    stats = []
    K = int(dvai_config.K)
    eta = float(dvai_config.impute_damping)

    for k in range(1, K + 1):
        print(f"\n========== DVAI iter {k}/{K} ==========")

        # ---- (M-step) diffusion train on completed X_{k-1} ----
        if bool(dvai_config.reset_diffusion_each_iter):
            imputer.reset()

        X_train_std = to_std(X_current[:, 1:])   # (N, D-1) standardized + EDM_SCALE
        diff_hist = imputer.fit(X_train_std, verbose=True)
        diff_last_loss = float(diff_hist["loss"][-1]) if len(diff_hist["loss"]) > 0 else np.nan

        # ---- (E-step) conditional imputation with observed anchored ----
        X_cond = X_current[:, 1:].copy().astype(np.float32)
        obs_pos = (~mask_feat)
        X_cond[obs_pos] = X_obs_feat[obs_pos]  # anchor observed

        X_cond_std = to_std(X_cond)
        if bool(diff_config.fill_missing_with_zero):
            X_cond_std[mask_feat] = 0.0

        X_imp_std = imputer.impute(X_obs=X_cond_std, obs_mask=obs_mask_feat)
        X_imp = from_std(X_imp_std)

        # 只更新缺失位置 + damping
        X_new = X_current[:, 1:].copy().astype(np.float32)
        X_new[mask_feat] = (1.0 - eta) * X_new[mask_feat] + eta * X_imp[mask_feat]

        # 观测位置强制回原始观测
        X_new[obs_pos] = X_obs_feat[obs_pos]

        X_current[:, 1:] = X_new
        force_intercept(X_current)

        miss_true = X_full[:, 1:][mask_feat]
        miss_est = X_current[:, 1:][mask_feat]
        im_mse = mse(miss_est, miss_true)
        im_rmse = rmse(miss_est, miss_true)
        im_mae = mae(miss_est, miss_true)
        print(f"[Impute] MSE={im_mse:.6f}, RMSE={im_rmse:.6f}, MAE={im_mae:.6f}")

        # ---- (VI-step) VB with prior = previous posterior ----
        vb_cfg_k = FullCovGVBConfig(
            lr=fullcov_vb_config.lr,
            max_iter=int(dvai_config.vb_max_iter_inner),
            num_mc=int(dvai_config.vb_num_mc_inner),
            grad_clip=fullcov_vb_config.grad_clip,
            window=fullcov_vb_config.window,
            seed=fullcov_vb_config.seed,
            verbose=fullcov_vb_config.verbose,
        )

        Xk_t = torch.tensor(X_current, device=device, dtype=torch.float32)

        if bool(dvai_config.use_posterior_as_prior):
            blend = float(dvai_config.prior_blend)
            cov_prev_t = torch.tensor(cov_prev, device=device, dtype=torch.float32)
            mu_prev_t = torch.tensor(mu_prev, device=device, dtype=torch.float32)
            base_cov = float(data_config.prior_var) * torch.eye(D, device=device, dtype=torch.float32)

            prior_cov = blend * cov_prev_t + (1.0 - blend) * base_cov
            # IMPORTANT: do NOT shrink the prior mean toward 0 when blending.
            prior_mean = mu_prev_t

            model_k = BayesianLogisticRegression(
                Xk_t, y_t,
                prior_var=data_config.prior_var,
                prior_mean=prior_mean,
                prior_cov=prior_cov,
                prior_jitter=float(dvai_config.prior_jitter),
            )
        else:
            model_k = BayesianLogisticRegression(Xk_t, y_t, prior_var=data_config.prior_var)

        vb_k = FullCovGaussianVB(dim=D, cfg=vb_cfg_k, device=device)
        with torch.no_grad():
            vb_k.mu.copy_(torch.tensor(mu_prev, device=device, dtype=torch.float32))

        out_k = vb_k.fit(model_k)
        mu_k, cov_k = out_k["mu"], out_k["cov"]
        elbo_k = out_k["elbo_history"]

        param_rmse_true = rmse(mu_k, theta_true)
        param_rmse_full = rmse(mu_k, mu_full)
        delta_mu = rmse(mu_k, mu_prev)

        var_k = np.diag(cov_k)
        var_full = np.diag(cov_full)
        kl_diag = kl_diag_gaussians(mu_k, var_k, mu_full, var_full)

        print(f"[Param] RMSE(mu,true)={param_rmse_true:.6f}, RMSE(mu,fullVB)={param_rmse_full:.6f}, "
              f"delta_mu={delta_mu:.6f}, KLdiag(dvai||full)={kl_diag:.3f}")

        if bool(dvai_config.save_every_iter):
            np.save(dvai_dir / f"X_iter_{k}.npy", X_current)
            np.save(dvai_dir / f"mu_iter_{k}.npy", mu_k)
            np.save(dvai_dir / f"cov_iter_{k}.npy", cov_k)
            np.save(dvai_dir / f"elbo_iter_{k}.npy", np.asarray(elbo_k, dtype=np.float32))

            save_param_table(res_dir / f"params_iter_{k}.csv", mu_k, cov_k, mu_full, cov_full)
            plot_posterior_dvai_comparison(
                theta_true=theta_true,
                mu_full=mu_full, cov_full=cov_full,
                mu_dvai=mu_k, cov_dvai=cov_k,
                elbo_history=elbo_k,
                save_path=str(res_dir / f"dvai_iter_{k}_posterior.png"),
                max_dim=8,
            )

        stats.append({
            "iter": k,
            "diff_loss_last": diff_last_loss,
            "impute_mse": im_mse,
            "impute_rmse": im_rmse,
            "impute_mae": im_mae,
            "param_rmse_true": param_rmse_true,
            "param_rmse_full": param_rmse_full,
            "delta_mu": delta_mu,
            "kl_diag_dvai_full": kl_diag,
            "elbo_last": float(elbo_k[-1]) if len(elbo_k) > 0 else np.nan,
        })

        mu_prev, cov_prev = mu_k, cov_k

        if delta_mu < float(dvai_config.convergence_tol):
            print(f"[DVAI] early stop: delta_mu < tol ({delta_mu:.6f} < {dvai_config.convergence_tol})")
            break

    # ======================
    # 7) save stats + curves
    # ======================
    stats_path = res_dir / "dvai_stats.csv"
    if len(stats) > 0:
        with open(stats_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(stats[0].keys()))
            w.writeheader()
            w.writerows(stats)
        print(f"[Stats] saved to: {stats_path}")

    if len(stats) > 0:
        import matplotlib.pyplot as plt
        it = [s["iter"] for s in stats]
        imr = [s["impute_rmse"] for s in stats]
        ima = [s["impute_mae"] for s in stats]
        prt = [s["param_rmse_true"] for s in stats]
        prf = [s["param_rmse_full"] for s in stats]

        plt.figure()
        plt.plot(it, imr, label="Impute RMSE")
        plt.plot(it, ima, label="Impute MAE")
        plt.xlabel("DVAI iter")
        plt.ylabel("Error (missing entries)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(res_dir / "dvai_impute_metrics.png", dpi=200)
        plt.close()

        plt.figure()
        plt.plot(it, prt, label="Param RMSE vs True")
        plt.plot(it, prf, label="Param RMSE vs Full-VB")
        plt.xlabel("DVAI iter")
        plt.ylabel("RMSE")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(res_dir / "dvai_param_metrics.png", dpi=200)
        plt.close()

        print(f"[Plot] saved to: {res_dir/'dvai_impute_metrics.png'}")
        print(f"[Plot] saved to: {res_dir/'dvai_param_metrics.png'}")


if __name__ == "__main__":
    main()
