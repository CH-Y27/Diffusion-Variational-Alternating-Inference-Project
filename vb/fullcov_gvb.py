# vb/fullcov_gvb.py
import numpy as np
import torch
import torch.nn as nn
from tqdm import trange


class FullCovGVBConfig:
    def __init__(self, lr=2e-3, max_iter=5000, num_mc=50, grad_clip=10.0, window=10, seed=42, verbose=200):
        self.lr = lr
        self.max_iter = max_iter
        self.num_mc = num_mc
        self.grad_clip = grad_clip
        self.window = window
        self.seed = seed
        self.verbose = verbose


class FullCovGaussianVB(nn.Module):
    """
    q(theta)=N(mu, Sigma), Sigma = L L^T, L lower-triangular with softplus diag.

    model must provide:
      log_joint(theta:[S,D]) -> [S]
    """

    def __init__(self, dim: int, cfg: FullCovGVBConfig, device: torch.device):
        super().__init__()
        self.dim = dim
        self.cfg = cfg
        self.device = device

        self.mu = nn.Parameter(torch.zeros(dim, device=device))
        self.L_unconstrained = nn.Parameter(torch.zeros(dim, dim, device=device))
        torch.manual_seed(cfg.seed)

    def _build_L(self):
        L = torch.tril(self.L_unconstrained)
        diag = torch.nn.functional.softplus(torch.diag(L)) + 1e-4
        L = L - torch.diag(torch.diag(L)) + torch.diag(diag)
        return L

    def sample_theta(self, S: int):
        L = self._build_L()
        eps = torch.randn(S, self.dim, device=self.device)
        theta = self.mu.unsqueeze(0) + eps @ L.t()
        return theta, L

    def log_q(self, theta: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        D = self.dim
        inv_Sigma = torch.cholesky_inverse(L)
        diff = theta - self.mu.unsqueeze(0)
        quad = torch.sum((diff @ inv_Sigma) * diff, dim=1)
        logdet = 2.0 * torch.log(torch.diag(L)).sum()
        log_q = -0.5 * (quad + logdet + D * torch.log(torch.tensor(2.0 * torch.pi, device=self.device)))
        return log_q

    def fit(self, model) -> dict:
        cfg = self.cfg
        opt = torch.optim.Adam(self.parameters(), lr=cfg.lr)

        elbo_hist = []
        best_elbo = -1e30
        best_state = None
        bad = 0

        pbar = trange(cfg.max_iter, desc="VB (FullCov-GVB)")
        for it in pbar:
            theta, L = self.sample_theta(cfg.num_mc)
            logp = model.log_joint(theta)          # [S]
            logq = self.log_q(theta, L)            # [S]
            elbo = (logp - logq).mean()

            loss = -elbo
            opt.zero_grad(set_to_none=True)
            loss.backward()

            # grad clip
            torch.nn.utils.clip_grad_norm_(self.parameters(), cfg.grad_clip)
            opt.step()

            elbo_val = float(elbo.detach().cpu().item())
            elbo_hist.append(elbo_val)

            # early stopping on smoothed elbo
            if it + 1 >= cfg.window:
                smooth = float(np.mean(elbo_hist[-cfg.window:]))
            else:
                smooth = elbo_val

            if smooth > best_elbo + 1e-6:
                best_elbo = smooth
                best_state = {k: v.detach().clone() for k, v in self.state_dict().items()}
                bad = 0
            else:
                bad += 1

            if cfg.verbose and (it + 1) % cfg.verbose == 0:
                pbar.set_postfix(elbo=elbo_val, smooth=smooth)

        if best_state is not None:
            self.load_state_dict(best_state)

        with torch.no_grad():
            L = self._build_L()
            cov = (L @ L.t()).detach().cpu().numpy()
            mu = self.mu.detach().cpu().numpy()

        # full smooth curve (aligned length)
        if len(elbo_hist) >= cfg.window:
            elbo_smooth = np.convolve(np.array(elbo_hist), np.ones(cfg.window) / cfg.window, mode="valid").tolist()
        else:
            elbo_smooth = []

        return {
            "mu": mu,
            "cov": cov,
            "elbo_history": elbo_hist,      # raw full length
            "elbo_smooth": elbo_smooth,     # length max_iter-window+1
        }
