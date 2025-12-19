# main_pre.py
import numpy as np
import torch
from pathlib import Path

from configs import data_config, fullcov_vb_config, mcmc_config
from untils.seed_device import set_global_seed, get_device
from untils.io_utils import ensure_dir, load_csv
from generate_miss.data_generation import generate_logistic_data
from tasks.logistic_regression import BayesianLogisticRegression
from vb.fullcov_gvb import FullCovGaussianVB, FullCovGVBConfig
from vb.mcmc_mala import run_mala
from visualization.plot_posteriors import plot_posterior_comparison


def _map_init_lbfgs(model: BayesianLogisticRegression, iters: int = 200):
    """
    MAP = argmax log_joint(theta)
    用 LBFGS 做稳定初始化。
    """
    device = model.X.device
    theta = torch.zeros(model.D, device=device, requires_grad=True)

    opt = torch.optim.LBFGS([theta], lr=1.0, max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad(set_to_none=True)
        # 关键修复：不要假设 log_joint 一定可索引 [0]
        loss = -model.log_joint(theta).mean()
        loss.backward()
        return loss

    opt.step(closure)
    return theta.detach()


def main():
    set_global_seed(data_config.seed)
    device = get_device()
    print(f"[Env] device = {device}")

    data_dir = Path("Data")
    ensure_dir(data_dir)

    X_path = data_dir / "X_full.csv"
    y_path = data_dir / "y.csv"
    theta_path = data_dir / "theta_true.csv"

    if X_path.exists() and y_path.exists() and theta_path.exists():
        print("[Data] loading csv ...")
        X = load_csv(X_path)
        y = load_csv(y_path).reshape(-1)
        theta_true = load_csv(theta_path).reshape(-1)
    else:
        print("[Data] generating logistic data ...")
        X, y, theta_true = generate_logistic_data(
            N=data_config.N,
            D=data_config.D,
            seed=data_config.seed,
            prior_var=data_config.prior_var,      # 与后续 DVAI 保持一致
            theta_scale=data_config.theta_scale,
            add_intercept=data_config.add_intercept,
            standardize=data_config.standardize,
            save_dir=data_dir,
        )

    print(f"[Data] N={X.shape[0]}, D={X.shape[1]}")

    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    y_t = torch.tensor(y, dtype=torch.float32, device=device)

    model = BayesianLogisticRegression(X_t, y_t, prior_var=data_config.prior_var)

    print("[Init] MAP (LBFGS) ...")
    theta_map = _map_init_lbfgs(model, iters=200)

    # ===== VB =====
    print("[VB] FullCov-GVB ...")
    vb_cfg = FullCovGVBConfig(
        lr=fullcov_vb_config.lr,
        max_iter=fullcov_vb_config.max_iter,
        num_mc=fullcov_vb_config.num_mc,
        grad_clip=fullcov_vb_config.grad_clip,
        window=fullcov_vb_config.window,
        seed=fullcov_vb_config.seed,
        verbose=fullcov_vb_config.verbose,
    )
    vb = FullCovGaussianVB(dim=model.D, cfg=vb_cfg, device=device)

    # MAP 初始化均值（通常能显著改善收敛与方差估计）
    with torch.no_grad():
        vb.mu.copy_(theta_map)

    vb_post = vb.fit(model)
    vb_mu = vb_post["mu"]
    vb_cov = vb_post["cov"]
    elbo_history = vb_post["elbo_history"]

    # ===== MCMC (MALA) =====
    print("[MCMC] mala ...")
    mcmc_samples = run_mala(
        model=model,
        num_mcmc=mcmc_config.num_mcmc,
        burnin=mcmc_config.burnin,
        thin=mcmc_config.thin,
        step_init=mcmc_config.step_init,
        target_acc=mcmc_config.target_acc,
        adapt_until=mcmc_config.adapt_until,
        seed=mcmc_config.seed,
        verbose=mcmc_config.verbose,
    )

    # ===== sanity (log_joint) =====
    with torch.no_grad():
        def lj(th):
            th = torch.tensor(th, device=device, dtype=torch.float32)
            return float(model.log_joint(th).mean().item())  # 关键修复

        logp_true = lj(theta_true)
        logp_map = float(model.log_joint(theta_map).mean().item())
        logp_vb = lj(vb_mu)
        logp_mcmc = lj(mcmc_samples.mean(axis=0))

    print("\n[Sanity] log_joint:")
    print(f"  true  = {logp_true:.3f}")
    print(f"  map   = {logp_map:.3f}")
    print(f"  vb    = {logp_vb:.3f}")
    print(f"  mcmc  = {logp_mcmc:.3f}")

    # ===== RMSE =====
    def rmse(a, b):
        a = np.asarray(a).reshape(-1)
        b = np.asarray(b).reshape(-1)
        return float(np.sqrt(np.mean((a - b) ** 2)))

    print("\n[RMSE]")
    print(f"  RMSE(VB, True)   = {rmse(vb_mu, theta_true):.6f}")
    print(f"  RMSE(MCMC, True) = {rmse(mcmc_samples.mean(axis=0), theta_true):.6f}")
    print(f"  RMSE(VB, MCMC)   = {rmse(vb_mu, mcmc_samples.mean(axis=0)):.6f}")

    # ===== Plot =====
    out_dir = Path("Results")
    ensure_dir(out_dir)
    save_path = out_dir / "posterior_compare_complete_logistic.png"

    plot_posterior_comparison(
        theta_true=theta_true,
        vb_mu=vb_mu,
        vb_cov=vb_cov,
        mcmc_samples=mcmc_samples,
        elbo_history=elbo_history,
        save_path=str(save_path),
        max_dim=8,
    )
    print(f"[Plot] saved to: {save_path}")


if __name__ == "__main__":
    main()
