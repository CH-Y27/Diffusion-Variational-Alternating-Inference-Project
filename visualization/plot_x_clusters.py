# visualization/plot_x_clusters.py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from sklearn.cluster import KMeans

# ============================================================
# 关键：获取项目根目录 ROOT（保证 data/ 能被找到）
# ============================================================
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data"


def load_x_data():
    """从 PROJECT_ROOT/data/X_full.csv 读取数据"""
    x_path = DATA_DIR / "X_full.csv"
    if not x_path.exists():
        raise FileNotFoundError(f"未找到 {x_path}，请先生成完整数据。")
    return np.loadtxt(x_path, delimiter=",")


def cluster_x(X, n_clusters=2, seed=42):
    """使用前四维做 KMeans 聚类用于着色"""
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init="auto")
    return kmeans.fit_predict(X[:, :4])


def plot_pairwise_scatter(X, labels, save_path):
    """绘制前4维的4×4散点矩阵（仅 scatter）"""
    dim = 4
    feature_names = [f"x[{i}]" for i in range(dim)]
    colors = ["green", "blue"]
    alpha = 0.5

    fig, axes = plt.subplots(dim, dim, figsize=(12, 12))

    for i in range(dim):
        for j in range(dim):
            ax = axes[i, j]

            # 两簇散点
            for k in range(2):
                ax.scatter(
                    X[labels == k, j],
                    X[labels == k, i],
                    s=8,
                    color=colors[k],
                    alpha=alpha
                )

            # 坐标轴设置（只在最边缘显示标签）
            if i == dim - 1:
                ax.set_xlabel(feature_names[j])
            else:
                ax.set_xticks([])

            if j == 0:
                ax.set_ylabel(feature_names[i])
            else:
                ax.set_yticks([])

            ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def main():
    # 1. 加载数据
    X = load_x_data()

    # 2. 聚类用于着色
    labels = cluster_x(X)

    # 3. 输出路径
    save_path = ROOT / "results" / "x_pairwise_scatter.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # 4. 绘制 4×4 scatter matrix
    plot_pairwise_scatter(X, labels, save_path)

    print(f"[Viz] 前4维散点矩阵已保存至：{save_path}")


if __name__ == "__main__":
    main()
