import numpy as np
import sys
import os

def print_npy_file(file_path):
    """
    打印npy文件的内容

    Args:
        file_path: .npy文件路径
        precision: 浮点数打印精度
        max_items: 显示的最大元素数量
    """
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误: 文件 '{file_path}' 不存在")
        return

    # 加载npy文件
    try:
        data = np.load(file_path)
    except Exception as e:
        print(f"错误: 无法加载文件 '{file_path}', 原因: {e}")
        return
    # 打印数组内容
    print("数组内容:")
    print(data)

if __name__ == "__main__":

    file_path = '/home/hdd1/sunao/ACMDM/frame_label/13.npy'
    print_npy_file(file_path)
