# tasks/logistic_regression.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class BayesianLogisticRegression(nn.Module):
    """
    y_i ~ Bernoulli(sigmoid(x_i^T theta))

    prior:
      - isotropic:  N(0, prior_var I)
      - full-cov:   N(prior_mean, prior_cov)

    log_joint(theta) -> [S]
    """

    def __init__(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        prior_var: float = 50.0,
        prior_mean: torch.Tensor | None = None,   # [D]
        prior_cov: torch.Tensor | None = None,    # [D,D]
        prior_jitter: float = 1e-4,
    ):
        super().__init__()
        assert X.ndim == 2
        assert y.ndim == 1
        self.X = X
        self.y = y
        self.N, self.D = X.shape

        self.prior_var = float(prior_var)

        # --- decide prior mode ---
        self.use_full_prior = (prior_mean is not None) and (prior_cov is not None)

        if self.use_full_prior:
            assert prior_mean.ndim == 1 and prior_mean.shape[0] == self.D
            assert prior_cov.ndim == 2 and prior_cov.shape[0] == self.D and prior_cov.shape[1] == self.D

            self.prior_mean = prior_mean.detach()
            cov = prior_cov.detach()

            # jitter for SPD
            cov = cov + float(prior_jitter) * torch.eye(self.D, device=cov.device, dtype=cov.dtype)

            # Cholesky for stable quad + logdet
            self.prior_L = torch.linalg.cholesky(cov)  # cov = L L^T
            self.prior_logdet = 2.0 * torch.log(torch.diag(self.prior_L)).sum()
        else:
            self.prior_mean = None
            self.prior_L = None
            self.prior_logdet = None

    def log_joint(self, theta: torch.Tensor) -> torch.Tensor:
        """
        theta: [D] or [S, D]
        return: [S]
        """
        if theta.ndim == 1:
            theta = theta.unsqueeze(0)
        assert theta.ndim == 2 and theta.shape[1] == self.D

        logits = self.X @ theta.t()      # [N, S]
        y = self.y.unsqueeze(1)          # [N, 1]

        loglik = y * F.logsigmoid(logits) + (1.0 - y) * F.logsigmoid(-logits)
        loglik = loglik.sum(dim=0)       # [S]

        # --- prior ---
        if not self.use_full_prior:
            quad = (theta ** 2).sum(dim=1)  # [S]
            const = self.D * torch.log(torch.tensor(2.0 * torch.pi * self.prior_var, device=theta.device))
            logprior = -0.5 * (quad / self.prior_var + const)
        else:
            diff = theta - self.prior_mean.unsqueeze(0)      # [S,D]
            # solve cov^{-1} diff^T via cholesky_solve
            sol = torch.cholesky_solve(diff.t(), self.prior_L)  # [D,S]
            quad = (diff * sol.t()).sum(dim=1)                 # [S]
            const = self.D * torch.log(torch.tensor(2.0 * torch.pi, device=theta.device))
            logprior = -0.5 * (quad + self.prior_logdet + const)

        return loglik + logprior
