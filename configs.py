# configs.py

# =========================
# Data
# =========================
class DataConfig:
    seed = 42
    N = 2000
    D = 100              # 不含截距列；add_intercept=True 则最终维度为 D+1
    add_intercept = True
    standardize = True

    # base prior (complete-data baseline & DVAI iter0)
    prior_var = 50.0
    theta_scale = 1.0

    # missing
    missing_rate = 0.30
    missing_seed = 123


# =========================
# VB (FullCov-GVB)
# =========================
class FullCovVBConfig:
    lr = 0.002
    max_iter = 5000
    num_mc = 50
    grad_clip = 10.0
    window = 10
    seed = 42
    verbose = 200


# =========================
# MCMC (for main_pre.py baseline)
# =========================
class MCMCConfig:
    num_mcmc = 100000
    burnin = 0.25
    thin = 10

    step_init = 1e-3
    target_acc = 0.57
    adapt_until = 20000

    seed = 42
    verbose = 5000


# =========================
# Diffusion
# =========================
class DiffusionConfig:
    # --- preprocessing (DiffPuter-style) ---
    standardize_from_observed = True
    fill_missing_with_zero = True
    clamp_val = 6.0

    # --- DDPM schedule (kept for compatibility; EDM impl doesn't use these) ---
    T = 200
    beta_start = 1e-4
    beta_end = 2e-2

    # --- MLP denoiser ---
    # NOTE: keep the network small for CPU-debug. You can scale up on server later.
    hidden = 64
    n_layers = 3
    t_embed_dim = 64
    dropout = 0.0

    # --- training ---
    lr = 2e-4
    batch_size = 256
    # NOTE: 2 epochs is effectively "not trained" for diffusion.
    # Use a larger value so the denoiser actually learns something on CPU.
    epochs = 120
    grad_clip = 1.0
    ema = 0.999

    # --- sampling ---
    # Fewer steps for CPU-debug; increase later on server.
    sample_steps = 30
    # Monte-Carlo average reduces variance of imputation.
    num_impute_samples = 8


# =========================
# DVAI outer loop
# =========================
class DVAIConfig:
    K = 5
    convergence_tol = 1e-4

    # Start conservative: if diffusion is still weak, large damping injects noise and worsens over iterations.
    # You can increase it after you observe imputation RMSE starts decreasing.
    impute_damping = 0.1

    vb_max_iter_inner = 5000
    vb_num_mc_inner = 30

    reset_diffusion_each_iter = False
    save_every_iter = True

    # your core innovation: posterior-as-prior
    use_posterior_as_prior = True
    prior_blend = 1.0
    prior_jitter = 1e-4


# =========================
# Instances (names must match main_pre.py imports)
# =========================
data_config = DataConfig()
fullcov_vb_config = FullCovVBConfig()
mcmc_config = MCMCConfig()

diff_config = DiffusionConfig()
dvai_config = DVAIConfig()
