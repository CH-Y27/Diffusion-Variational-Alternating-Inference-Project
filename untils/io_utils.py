# untils/io_utils.py
import numpy as np
from pathlib import Path


def ensure_dir(path: str | Path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_csv(path: str | Path, arr: np.ndarray):
    """
    正确签名：save_csv(path, array)
    """
    path = Path(path)
    ensure_dir(path.parent)
    np.savetxt(path, arr, delimiter=",")


def load_csv(path: str | Path) -> np.ndarray:
    path = Path(path)
    return np.loadtxt(path, delimiter=",")
