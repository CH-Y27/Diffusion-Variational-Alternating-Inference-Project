# configs.py

# =========================
# Data
# =========================
class DataConfig:
    seed = 42
    N = 2000
    D = 100              # 不含截距列；若 add_intercept=True，则最终 X 维度为 D+1
    add_intercept = True

    # 是否在生成/训练时做标准化（取决于你的 data_generation 实现）
    standardize = True

    # logistic prior: Normal(0, prior_var)
    prior_var = 50.0

    # 控制生成难度（logit尺度）
    theta_scale = 1.0

    # ===== 缺失相关（统一由 configs 管理）=====
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
# MCMC
# =========================
class MCMCConfig:
    num_mcmc = 100000
    burnin = 0.25
    thin = 10

    # 如果你是 MALA/自适应步长，会用到这些
    step_init = 1e-3
    target_acc = 0.57
    adapt_until = 20000

    seed = 42
    verbose = 5000


# =========================
# Diffusion (for imputation)
# ⚠️ 字段名必须与 models/diffusion_imputer.py::DiffusionImputerConfig 完全一致
# =========================
class DiffusionConfig:
    # schedule
    T = 500
    beta_start = 1e-4
    beta_end = 2e-2

    # network (你的 imputer 读取 hidden / n_layers / t_embed_dim)
    hidden = 256
    n_layers = 4
    t_embed_dim = 128
    dropout = 0.0

    # training
    lr = 2e-4
    batch_size = 256
    epochs = 10
    grad_clip = 1.0
    ema = 0.999

    # sampling
    sample_steps = 200
    num_impute_samples = 8
    clamp_val = 6.0

    # ---- 兼容别名（如果你别处写了这些名字，也不会炸）----
    @property
    def hidden_dim(self):
        return self.hidden

    @property
    def num_layers(self):
        return self.n_layers

    @property
    def impute_samples(self):
        return self.num_impute_samples


# =========================
# DVAI outer loop
# =========================
class DVAIConfig:
    K = 5
    convergence_tol = 1e-3

    # 每轮扩散训练迭代数（main.py 若用 epochs/steps 以外的控制变量）
    diffusion_train_iters = 150

    # 阻尼：防止每轮填补导致 VB 后验跑远
    impute_damping = 0.2

    # 填补时的采样次数（用于均值稳定）
    diff_samples = 10


# =========================
# Instances
# =========================
data_config = DataConfig()
fullcov_vb_config = FullCovVBConfig()
mcmc_config = MCMCConfig()
diff_config = DiffusionConfig()
dvai_config = DVAIConfig()
