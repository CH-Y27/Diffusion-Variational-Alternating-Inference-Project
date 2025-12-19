# vb/mcmc_mala.py
import numpy as np
import torch
from tqdm import trange


def run_mala(
    model,
    num_mcmc: int = 100000,
    burnin: float = 0.25,
    thin: int = 10,
    step_init: float = 1e-3,
    target_acc: float = 0.57,
    adapt_until: int = 20000,
    seed: int = 42,
    verbose: int = 5000,
):
    """
    MALA with MH correction.
    model must provide log_joint(theta:[S,D] or [D]) -> [S] (或标量也能处理)
    """
    torch.manual_seed(seed)
    device = model.X.device
    D = model.D

    theta = torch.zeros(D, device=device)

    def logp_and_grad(th: torch.Tensor):
        th = th.detach().requires_grad_(True)
        lp_vec = model.log_joint(th)
        lp = lp_vec.mean()  # 关键：不再用 [0]
        g = torch.autograd.grad(lp, th)[0]
        return lp.detach(), g.detach()

    logp, grad = logp_and_grad(theta)

    step = float(step_init)
    acc_count = 0
    kept = []

    pbar = trange(num_mcmc, desc="MCMC-MALA")
    for it in pbar:
        noise = torch.randn(D, device=device)
        mean_prop = theta + 0.5 * (step ** 2) * grad
        theta_prop = mean_prop + step * noise

        logp_prop, grad_prop = logp_and_grad(theta_prop)

        def log_q(a, b, grad_b):
            m = b + 0.5 * (step ** 2) * grad_b
            diff = a - m
            return -0.5 * (diff @ diff) / (step ** 2)

        logq_fwd = log_q(theta_prop, theta, grad)
        logq_bwd = log_q(theta, theta_prop, grad_prop)

        log_alpha = (logp_prop + logq_bwd) - (logp + logq_fwd)
        if torch.log(torch.rand((), device=device)) < log_alpha:
            theta = theta_prop
            logp = logp_prop
            grad = grad_prop
            acc_count += 1

        if it < adapt_until:
            acc_rate = acc_count / (it + 1)
            step = step * np.exp((acc_rate - target_acc) / np.sqrt(it + 1))

        if verbose and (it + 1) % verbose == 0:
            acc_rate = acc_count / (it + 1)
            pbar.write(f"[MALA] iter {it+1}/{num_mcmc}, acc={acc_rate:.3f}, step={step:.2e}")

        if it >= int(num_mcmc * burnin) and ((it - int(num_mcmc * burnin)) % thin == 0):
            kept.append(theta.detach().cpu().numpy())

    samples = np.stack(kept, axis=0) if len(kept) > 0 else np.zeros((0, D))
    return samples
