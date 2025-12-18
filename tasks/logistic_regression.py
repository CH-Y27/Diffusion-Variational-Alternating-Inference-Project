# tasks/logistic_regression.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class BayesianLogisticRegression(nn.Module):
    """
    y_i ~ Bernoulli(sigmoid(x_i^T theta))
    theta ~ N(0, prior_var * I)

    提供：
      log_joint(theta)  -> [S]
      grad_log_joint(theta) 可由 autograd 在 MALA 内部得到
    """

    def __init__(self, X: torch.Tensor, y: torch.Tensor, prior_var: float = 50.0):
        super().__init__()
        assert X.ndim == 2
        assert y.ndim == 1
        self.X = X
        self.y = y
        self.N, self.D = X.shape
        self.prior_var = float(prior_var)

    def log_joint(self, theta: torch.Tensor) -> torch.Tensor:
        """
        theta: [D] or [S, D]
        return: [S]
        """
        if theta.ndim == 1:
            theta = theta.unsqueeze(0)
        assert theta.ndim == 2 and theta.shape[1] == self.D

        logits = self.X @ theta.t()  # [N, S]
        y = self.y.unsqueeze(1)      # [N, 1]

        # stable log-likelihood:
        # log p(y|logits) = y*logsigmoid(logits) + (1-y)*logsigmoid(-logits)
        loglik = y * F.logsigmoid(logits) + (1.0 - y) * F.logsigmoid(-logits)
        loglik = loglik.sum(dim=0)  # [S]

        # Gaussian prior N(0, prior_var I)
        # log p(theta) = -0.5*(||theta||^2/prior_var + D*log(2π prior_var))
        quad = (theta ** 2).sum(dim=1)  # [S]
        logprior = -0.5 * (quad / self.prior_var + self.D * torch.log(torch.tensor(2.0 * torch.pi * self.prior_var, device=theta.device)))

        return loglik + logprior
