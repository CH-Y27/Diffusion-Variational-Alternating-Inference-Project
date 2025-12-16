# data_generate_miss.py  （项目根目录）

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GEN_DIR = ROOT / "generate_miss"
DATA_DIR = ROOT / "data"

for p in (ROOT, GEN_DIR, DATA_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from configs import data_config
from data_generation import generate_logistic_data
from data_miss import apply_mcar_missing, save_mcar_outputs


def main():
    # 1) 生成完整数据（内部根据 configs 保存到 data_dir）
    X_full, y, theta_true = generate_logistic_data()

    # 2) 生成缺失 + 初始填补（内部根据 configs 的 missing_rate/missing_seed）
    X_miss, mask_m, X_init = apply_mcar_missing(X_full)

    # 3) 保存缺失文件（内部根据 configs 保存到 data_dir）
    save_mcar_outputs(X_miss=X_miss, mask=mask_m, X_init=X_init)

    print(f"[Done] Saved all files to: {data_config.data_dir}")


if __name__ == "__main__":
    main()
