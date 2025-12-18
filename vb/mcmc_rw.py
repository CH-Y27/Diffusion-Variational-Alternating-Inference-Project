# vb/mcmc_rw.py
import numpy as np
import torch
from tqdm import trange

def run_rwm(model, cfg, theta_init=None, seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = model.X.device
    D = model.D

    n_iter = cfg.n_iter
    burn_in = int(cfg.burn_in_rate * n_iter)
    target = cfg.target_accept
    num_cov = cfg.num_covariance
    sig_scale = cfg.sig_scale
    scale = cfg.init_scale
    thin = cfg.thin
    verbose = cfg.verbose

    theta = torch.zeros(D, device=device) if theta_init is None else theta_init.detach().clone().to(device)
    logp = model.log_joint(theta)[0].item()

    V = sig_scale * np.eye(D, dtype=np.float64)
    all_samples = np.zeros((n_iter, D), dtype=np.float64)
    acc = 0
    window = []

    for it in trange(n_iter, desc="MCMC-RWM"):
        L = np.linalg.cholesky(V + 1e-8 * np.eye(D))
        delta = scale * (L @ np.random.randn(D))
        prop = theta + torch.tensor(delta, dtype=theta.dtype, device=device)

        logp_prop = model.log_joint(prop)[0].item()
        if np.log(np.random.rand()) < (logp_prop - logp):
            theta = prop
            logp = logp_prop
            accepted = 1
        else:
            accepted = 0

        acc += accepted
        all_samples[it] = theta.detach().cpu().numpy()

        window.append(all_samples[it].copy())
        if len(window) > num_cov:
            window.pop(0)

        if it > 50:
            gamma = 1.0 / np.sqrt(it)
            scale = float(np.exp(np.log(scale) + gamma * (accepted - target)))
            W = np.array(window)
            if W.shape[0] > 2:
                V = np.cov(W.T) + 1e-6 * np.eye(D)

        if verbose and ((it + 1) % verbose == 0):
            print(f"[RWM] iter {it+1}/{n_iter}, acc={acc/(it+1):.3f}, scale={scale:.2e}")

    kept = all_samples[burn_in::thin]
    return kept
