# visualization/plot_posteriors.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, norm

def plot_posterior_comparison(theta_true, vb_mu, vb_cov, mcmc_samples, elbo_history, save_path, max_dim=8):
    theta_true = np.asarray(theta_true).reshape(-1)
    vb_mu = np.asarray(vb_mu).reshape(-1)
    vb_cov = np.asarray(vb_cov)
    mcmc_samples = np.asarray(mcmc_samples)

    D = vb_mu.shape[0]
    K = min(max_dim, D)

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.flatten()

    for i in range(K):
        ax = axes[i]
        std = float(np.sqrt(max(vb_cov[i, i], 1e-12)))
        xs = np.linspace(vb_mu[i] - 4 * std, vb_mu[i] + 4 * std, 400)

        ys_vb = norm.pdf(xs, vb_mu[i], std)
        ax.plot(xs, ys_vb, color="red", lw=1.5, label="VB")
        ax.fill_between(xs, ys_vb, alpha=0.15, color="red")

        try:
            kde = gaussian_kde(mcmc_samples[:, i])
            ys_mcmc = kde(xs)
            ax.plot(xs, ys_mcmc, color="green", lw=1.5, label="MCMC KDE")
            ax.fill_between(xs, ys_mcmc, alpha=0.15, color="green")
        except Exception:
            pass

        ax.axvline(theta_true[i], color="black", ls="--", lw=1.2, label="True" if i == 0 else None)
        ax.set_title(rf"$\theta_{i}$")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend()

    ax = axes[8]
    ax.plot(elbo_history, color="purple", lw=1.2)
    ax.set_title("ELBO Curve")
    ax.set_xlabel("Iteration")
    ax.grid(True, alpha=0.3)

    for j in range(K, 8):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)
