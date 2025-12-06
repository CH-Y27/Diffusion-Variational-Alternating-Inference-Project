# vb/analytic_posterior.py
import numpy as np
import torch

def analytic_posterior_linear(X: np.ndarray,
                              y: np.ndarray,
                              noise_var: float,
                              prior_var: float):
    """
    线性高斯模型的精确后验：
        θ | X, y ~ N(μ_post, Σ_post)

    返回：
        mu_post: [D]
        Sigma_post: [D, D]
    """
    X_t = torch.tensor(X, dtype=torch.float64)
    y_t = torch.tensor(y, dtype=torch.float64).reshape(-1, 1)

    N, D = X_t.shape

    XtX = X_t.t() @ X_t              # [D, D]
    Xty = X_t.t() @ y_t              # [D, 1]

    Sigma_inv = XtX / noise_var + torch.eye(D, dtype=torch.float64) / prior_var
    Sigma_post = torch.inverse(Sigma_inv)
    mu_post = Sigma_post @ (Xty / noise_var)

    return mu_post.squeeze(1).numpy(), Sigma_post.numpy()
