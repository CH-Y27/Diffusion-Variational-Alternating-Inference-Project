# tasks/linear_regression.py
import torch
import torch.nn as nn


class BayesianLinearRegression(nn.Module):
    """
    线性高斯回归：
        y | x, θ ~ N(x^T θ, σ^2)
        θ ~ N(μ0, Σ0)

    这里只在 log_joint 中用到先验的二次型部分，
    常数项 log|Σ0| 等对 ELBO 的梯度是常数，可忽略。
    """

    def __init__(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        noise_var: float = 0.3 ** 2,
        prior_var: float = 1.0,
        prior_mean: torch.Tensor | None = None,
        prior_prec: torch.Tensor | None = None,
    ):
        """
        Parameters
        ----------
        X : [N, D]
        y : [N]
        noise_var : σ^2
        prior_var : 若不给 prior_prec，则使用 prior_var * I 作为协方差
        prior_mean : [D]，先验均值 μ0，可选
        prior_prec : [D, D]，先验精度矩阵 Σ0^{-1}，可选
        """
        super().__init__()
        assert X.dim() == 2
        assert y.dim() == 1 and y.shape[0] == X.shape[0]

        self.X = X
        self.y = y
        self.N, self.D = X.shape
        self.noise_var = float(noise_var)

        device = X.device
        dtype = X.dtype

        # 先验均值
        if prior_mean is None:
            self.prior_mean = torch.zeros(self.D, device=device, dtype=dtype)
        else:
            self.prior_mean = prior_mean.to(device=device, dtype=dtype)
            assert self.prior_mean.shape == (self.D,)

        # 先验精度矩阵 Σ0^{-1}
        if prior_prec is None:
            prec_scalar = 1.0 / float(prior_var)
            self.prior_prec = torch.eye(self.D, device=device, dtype=dtype) * prec_scalar
        else:
            self.prior_prec = prior_prec.to(device=device, dtype=dtype)
            assert self.prior_prec.shape == (self.D, self.D)

    # ------------------------------------------------
    # log p(y, θ) = log p(θ) + log p(y | θ)（忽略常数）
    # θ 输入可以是 [D] 或 [S, D]，输出对应 [S]
    # ------------------------------------------------
    def log_joint(self, theta: torch.Tensor) -> torch.Tensor:
        if theta.dim() == 1:
            theta = theta.unsqueeze(0)  # [1, D]
        assert theta.shape[1] == self.D

        # 先验项：-1/2 (θ-μ0)^T Σ0^{-1} (θ-μ0)
        diff = theta - self.prior_mean  # [S, D]
        prior_term = -0.5 * torch.einsum("bi,ij,bj->b", diff, self.prior_prec, diff)

        # 似然项：-1/(2σ^2) ||y - Xθ||^2
        # y_pred: [N, S]
        y_pred = self.X @ theta.T
        resid = self.y.view(-1, 1) - y_pred
        like_term = -0.5 / self.noise_var * torch.sum(resid ** 2, dim=0)  # [S]

        return prior_term + like_term

    # （可选）只要似然
    def log_likelihood(self, theta: torch.Tensor) -> torch.Tensor:
        if theta.dim() == 1:
            theta = theta.unsqueeze(0)
        y_pred = self.X @ theta.T
        resid = self.y.view(-1, 1) - y_pred
        like_term = -0.5 / self.noise_var * torch.sum(resid ** 2, dim=0)
        return like_term
