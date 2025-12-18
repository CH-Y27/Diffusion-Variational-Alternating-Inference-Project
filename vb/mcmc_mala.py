# vb/mcmc_mala.py
import numpy as np
import torch
from tqdm import trange


@torch.no_grad()
def _gaussian_logpdf(x: torch.Tensor, mean: torch.Tensor, cov_diag: float):
    # Not used; kept minimal for clarity
    raise NotImplementedError


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
    model must provide log_joint(theta:[S,D]) -> [S]
    """
    torch.manual_seed(seed)
    device = model.X.device
    D = model.D

    # init at zero (you也可以传 MAP，这里保持接口简单)
    theta = torch.zeros(D, device=device)

    def logp_and_grad(th: torch.Tensor):
        th = th.detach().requires_grad_(True)
        lp = model.log_joint(th)[0]
        g = torch.autograd.grad(lp, th)[0]
        return lp.detach(), g.detach()

    logp, grad = logp_and_grad(theta)

    step = float(step_init)
    acc_count = 0

    kept = []
    total_keep = int((num_mcmc * (1 - burnin)) / thin)

    pbar = trange(num_mcmc, desc="MCMC-MALA")
    for it in pbar:
        # proposal: th' = th + 0.5*eps^2*grad + eps*N(0,I)
        noise = torch.randn(D, device=device)
        mean_prop = theta + 0.5 * (step ** 2) * grad
        theta_prop = mean_prop + step * noise

        logp_prop, grad_prop = logp_and_grad(theta_prop)

        # compute q(th | th') and q(th' | th) in log form
        # q(a|b) = N(a; b + 0.5 eps^2 grad(b), eps^2 I)
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

        # adapt step (Robbins–Monro on log step)
        if it < adapt_until:
            acc_rate = acc_count / (it + 1)
            step = step * np.exp((acc_rate - target_acc) / np.sqrt(it + 1))

        if verbose and (it + 1) % verbose == 0:
            acc_rate = acc_count / (it + 1)
            pbar.write(f"[MALA] iter {it+1}/{num_mcmc}, acc={acc_rate:.3f}, step={step:.2e}")

        # save
        if it >= int(num_mcmc * burnin) and ((it - int(num_mcmc * burnin)) % thin == 0):
            kept.append(theta.detach().cpu().numpy())

    samples = np.stack(kept, axis=0) if len(kept) > 0 else np.zeros((0, D))
    return samples
