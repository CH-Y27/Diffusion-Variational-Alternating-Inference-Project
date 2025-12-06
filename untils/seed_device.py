# untils/seed_device.py
import random
import numpy as np
import torch

def set_global_seed(seed: int = 42):
    """
    设定 Python / NumPy / PyTorch 的全局随机种子，保证实验可复现
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 确保 CuDNN 也尽量确定性（有 GPU 时）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """
    若存在 GPU 则使用 GPU, 否则自动回退到 CPU
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
