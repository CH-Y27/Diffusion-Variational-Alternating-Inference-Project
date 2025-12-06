# untils/io_utils.py
from pathlib import Path
import numpy as np
import pandas as pd

def ensure_dir(path: str | Path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def save_csv(array: np.ndarray, path: str | Path):
    path = Path(path)
    ensure_dir(path.parent)
    df = pd.DataFrame(array)
    df.to_csv(path, index=False)

def load_csv(path: str | Path) -> np.ndarray:
    return pd.read_csv(path).to_numpy()

def save_npy(array: np.ndarray, path: str | Path):
    path = Path(path)
    ensure_dir(path.parent)
    np.save(path, array)

def load_npy(path: str | Path) -> np.ndarray:
    return np.load(path)
