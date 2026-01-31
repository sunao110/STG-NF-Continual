import os

import numpy as np
import matplotlib.pyplot as plt
from setuptools.sandbox import save_path


def plot_score_curve(scores, file_name, offset=0, save_dir='/home/hdd1/sunao/ACMDM/figures', title="Score Curve",
                     xlabel="Frame Index", ylabel="Score", figsize=(12, 6),
                     dpi=100):
    """
    绘制得分曲线并保存到指定位置

    Args:
        scores: 一维numpy数组，每帧的得分
        save_path: 保存路径
        title: 图像标题
        xlabel: x轴标签
        ylabel: y轴标签
        figsize: 图像大小 (width, height)
        dpi: 图像分辨率
    """
    # 创建图形
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # 绘制曲线
    x_values = np.arange(len(scores)) + offset
    ax.plot(x_values, scores, linewidth=1.0)

    # 设置标题和轴标签
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # 添加网格
    ax.grid(True, alpha=0.3)

    # 保存图像
    plt.tight_layout()
    save_path = os.path.join(save_dir, file_name)
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)  # 关闭图形释放内存

    print(f"Score curve saved to: {save_path}")
