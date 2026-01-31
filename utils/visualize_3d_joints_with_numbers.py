import numpy as np
import matplotlib.pyplot as plt
# 移除了 from mpl_toolkits.mplot3d import Axes3D


def visualize_2d_joints_with_numbers(joints_data, title="2D Joint Visualization"):
    """
    可视化2D关节并标注节点序号

    Args:
        joints_data: 形状为 (22, 3) 的numpy数组，表示22个关节的坐标
        title: 图像标题
    """
    # 创建2D图形
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111)  # 移除了projection='3d'

    # 绘制关节点 - 只取x和y坐标
    x = joints_data[:, 0]
    y = joints_data[:, 1]
    # z = joints_data[:, 2]  # 忽略z坐标

    # 绘制散点
    ax.scatter(x, y, c='red', s=100, alpha=0.8)

    # 标注每个关节的序号
    for i in range(len(joints_data)):
        ax.text(x[i], y[i], f'{i}', fontsize=12, ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5))

    # 设置坐标轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    # 移除了ax.set_zlabel('Z')

    # 设置标题
    ax.set_title(title)

    # 移除了ax.view_init(elev=20, azim=45)

    plt.show()


if __name__ == "__main__":
    # 加载你的实际数据
    joints_data = np.load('/home/hdd1/sunao/ACMDM/train/101.npy')

    # 可视化
    visualize_2d_joints_with_numbers(joints_data[1], "22-Joint Human Skeleton Visualization (2D Projection)")
