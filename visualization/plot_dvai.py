# visualization/plot_dvai.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def plot_posterior_dvai_comparison(
    theta_true,
    mu_full,
    cov_full,
    mu_dvai,
    cov_dvai,
    elbo_history,
    save_path,
    max_dim=8
):
    """
    Plot:
      - Full-data VB posterior (blue)
      - DVAI VB posterior (red)
      - True theta (black dashed vertical line)
      - ELBO curve (DVAI VB last inner loop), if provided
    """
    theta_true = np.asarray(theta_true).reshape(-1)
    mu_full = np.asarray(mu_full).reshape(-1)
    mu_dvai = np.asarray(mu_dvai).reshape(-1)
    cov_full = np.asarray(cov_full)
    cov_dvai = np.asarray(cov_dvai)

    D = mu_full.shape[0]
    K = min(max_dim, D)

    nrows = 3
    ncols = 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 16))
    axes = axes.flatten()

    for i in range(K):
        ax = axes[i]
        s_full = float(np.sqrt(max(cov_full[i, i], 1e-12)))
        s_dvai = float(np.sqrt(max(cov_dvai[i, i], 1e-12)))

        left = min(mu_full[i] - 4 * s_full, mu_dvai[i] - 4 * s_dvai)
        right = max(mu_full[i] + 4 * s_full, mu_dvai[i] + 4 * s_dvai)
        xx = np.linspace(left, right, 400)

        yy_full = norm.pdf(xx, mu_full[i], s_full)
        yy_dvai = norm.pdf(xx, mu_dvai[i], s_dvai)

        ax.plot(xx, yy_full, label="Full-data VB")
        ax.fill_between(xx, 0, yy_full, alpha=0.2)

        ax.plot(xx, yy_dvai, label="DVAI VB")
        ax.fill_between(xx, 0, yy_dvai, alpha=0.2)

        ax.axvline(theta_true[i], linestyle="--", linewidth=1.5, label="True θ" if i == 0 else None)
        ax.set_title(r"$\theta_{{{}}}$".format(i))
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend()

    ax = axes[-1]
    ax.set_title("ELBO (DVAI VB)")
    if elbo_history is not None and len(elbo_history) > 0:
        ax.plot(elbo_history, linewidth=1.5)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Iteration")

    for j in range(K, nrows * ncols - 1):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)
