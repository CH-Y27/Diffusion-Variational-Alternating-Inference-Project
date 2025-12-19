# vb/fullcov_gvb.py
import numpy as np
import torch
import torch.nn as nn
from tqdm import trange


class FullCovGVBConfig:
    def __init__(
        self,
        lr=2e-3,
        max_iter=5000,
        num_mc=50,
        grad_clip=10.0,
        window=10,
        seed=42,
        verbose=200,
        # ====== 新增（可选）======
        patience=None,          # 例如 400；None 表示不早停
        restore_best=True,      # 训练结束恢复 best_state
        min_improve=1e-6,       # 判定“更好”的阈值
    ):
        self.lr = lr
        self.max_iter = max_iter
        self.num_mc = num_mc
        self.grad_clip = grad_clip
        self.window = window
        self.seed = seed
        self.verbose = verbose

        self.patience = patience
        self.restore_best = restore_best
        self.min_improve = min_improve


class FullCovGaussianVB(nn.Module):
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

    def fit(self, model):
        cfg = self.cfg
        opt = torch.optim.Adam(self.parameters(), lr=cfg.lr)

        elbo_hist = []
        best_state = None
        best_elbo = -1e18
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
            torch.nn.utils.clip_grad_norm_(self.parameters(), cfg.grad_clip)
            opt.step()

            elbo_val = float(elbo.detach().cpu().item())
            elbo_hist.append(elbo_val)

            if it + 1 >= cfg.window:
                smooth = float(np.mean(elbo_hist[-cfg.window:]))
            else:
                smooth = elbo_val

            if smooth > best_elbo + float(cfg.min_improve):
                best_elbo = smooth
                best_state = {k: v.detach().clone() for k, v in self.state_dict().items()}
                bad = 0
            else:
                bad += 1

            if cfg.verbose and (it % cfg.verbose == 0 or it == cfg.max_iter - 1):
                pbar.set_postfix(elbo=f"{elbo_val:.2e}", smooth=f"{smooth:.2e}", bad=bad)

            # ====== 可选早停 ======
            if cfg.patience is not None and bad >= int(cfg.patience):
                break

        # ====== 恢复最优点（非常关键：否则你可能拿到最后一步的坏点）======
        if cfg.restore_best and best_state is not None:
            self.load_state_dict(best_state, strict=True)

        with torch.no_grad():
            L = self._build_L()
            cov = (L @ L.t()).detach().cpu().numpy()
            mu = self.mu.detach().cpu().numpy()

        if len(elbo_hist) >= cfg.window:
            elbo_smooth = np.convolve(np.array(elbo_hist), np.ones(cfg.window) / cfg.window, mode="valid").tolist()
        else:
            elbo_smooth = []

        return {
            "mu": mu,
            "cov": cov,
            "elbo_history": elbo_hist,
            "elbo_smooth": elbo_smooth,
            "best_smooth": best_elbo,
        }
