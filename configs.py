# configs.py
from dataclasses import dataclass


# ... 保留原来的 DataConfig, MCMCConfig 等 ...

@dataclass
class FullCovVBConfig:
    n_iter: int = 10000
    mc_samples: int = 50
    lr: float = 1e-3
    grad_clip: float = 50.0
    lb_smooth: int = 50
    patience: int = 1000




@dataclass
class DataConfig:
    n_samples: int = 2000
    dim_x: int = 50
    noise_std: float = 0.3
    seed: int = 42
    data_dir: str = "data"

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
    # 这里我们用“稳一点”的设置
    n_iter: int = 3000          # 迭代多一点
    mc_samples: int = 20        # MC 样本数
    lr: float = 1e-3            # Adam 学习率
    prior_var: float = 1.0
    noise_var: float = 0.3 ** 2

@dataclass
class MCMCConfig:
    n_iter: int = 80000          # 总迭代次数（对应 NumMCMC）
    burn_in_rate: float = 0.3     # Burn-in 比例（对应 BurnInRate）
    target_accept: float = 0.25   # 目标接受率（TargetAccept）
    num_covariance: int = 1000    # 用最近多少个样本估计协方差（NumCovariance）
    sig_scale: float = 0.01       # 初始协方差尺度（SigScale）
    init_scale: float = 1.0       # 初始 proposal 缩放因子（Scale）
    thin: int = 1                 # thinning 步长（原文没细用，可以先设 1）
    verbose: int = 5000           # 每多少步打印一次进度


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
    max_outer_iter: int = 3
    convergence_tol: float = 1e-3

data_config = DataConfig()
vb_config = VBGaussConfig()
nagvac_config = NAGVACConfig()
mcmc_config = MCMCConfig()
diff_config = DiffusionConfig()
dvai_config = DVAIConfig()
fullcov_vb_config = FullCovVBConfig()