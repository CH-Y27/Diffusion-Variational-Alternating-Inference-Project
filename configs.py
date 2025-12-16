# configs.py
from dataclasses import dataclass


# ============================================================
# 1) Data Config (用于：生成完整数据 + 形成 MCAR 缺失)
# ============================================================

@dataclass
class DataConfig:
    # ----- dataset size -----
    n_samples: int = 2000
    dim_x: int = 100

    # ----- reproducibility -----
    seed: int = 42
    data_dir: str = "data"

    # ----- task type (固定 logistic) -----
    task: str = "logistic"
    add_intercept: bool = True

    # ----- X generation -----
    manifold: bool = True
    latent_dim: int = 20
    two_cluster: bool = True
    cluster_shift: float = 20.0
    embed_hidden: int = 128
    manifold_noise: float = 0.10

    # ----- theta generation -----
    theta0_std: float = 1.0
    theta_std: float = 1.0

    # ----- y generation (logistic) -----
    sigmoid_clip: float = 30.0  # clip logits 到 [-clip, clip] 做数值稳定

    # ----- MCAR missingness -----
    missing_rate: float = 0.30
    missing_seed: int = 123

    # ----- mean imputation -----
    impute_strategy: str = "col_mean"  # 目前只实现列均值


# ============================================================
# 2) 其他 configs 先保持不动（避免后续模块 import 断裂）
# ============================================================

@dataclass
class FullCovVBConfig:
    n_iter: int = 10000
    mc_samples: int = 50
    lr: float = 1e-3
    grad_clip: float = 50.0
    lb_smooth: int = 50
    patience: int = 1000


@dataclass
class VBGaussConfig:
    n_iter: int = 150
    mc_samples: int = 10
    lr: float = 1e-2
    init_scale: float = 0.1
    prior_var: float = 1.0
    noise_var: float = 0.3 ** 2


@dataclass
class NAGVACConfig:
    n_iter: int = 3000
    mc_samples: int = 20
    lr: float = 1e-3
    prior_var: float = 1.0
    noise_var: float = 0.3 ** 2


@dataclass
class MCMCConfig:
    n_iter: int = 80000
    burn_in_rate: float = 0.3
    target_accept: float = 0.25
    num_covariance: int = 1000
    sig_scale: float = 0.01
    init_scale: float = 1.0
    thin: int = 1
    verbose: int = 5000


@dataclass
class DiffusionConfig:
    T: float = 1.0
    n_steps: int = 50
    n_impute_samples: int = 10
    lr: float = 1e-4
    n_epochs: int = 50
    batch_size: int = 256


@dataclass
class DVAIConfig:
    max_outer_iter: int = 5
    convergence_tol: float = 1e-3


# ============================================================
# 3) Global instances
# ============================================================

data_config = DataConfig()
vb_config = VBGaussConfig()
nagvac_config = NAGVACConfig()
mcmc_config = MCMCConfig()
diff_config = DiffusionConfig()
dvai_config = DVAIConfig()
fullcov_vb_config = FullCovVBConfig()
