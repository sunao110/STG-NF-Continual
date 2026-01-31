import os
import numpy as np


def create_ones_from_npy_files(input_folder, output_folder):
    """
    读取输入文件夹下的所有.npy文件，生成与原文件第一个维度长度相同的1维全1数组，
    并保存到输出文件夹，文件名与原文件相同

    Args:
        input_folder (str): 输入文件夹路径
        output_folder (str): 输出文件夹路径
    """
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 遍历输入文件夹下的所有文件
    for filename in os.listdir(input_folder):
        # 检查是否为.npy文件
        if filename.lower().endswith('.npy'):
            # 构建完整的文件路径
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # 读取原始npy文件
            original_array = np.load(input_path)

            # 创建与原数组第一个维度长度相同的1维全1数组
            ones_array = np.ones(original_array.shape[0])

            # 保存新数组到输出文件夹
            np.save(output_path, ones_array)

            print(f"已处理: {filename} -> 原形状: {original_array.shape}, 新形状: {ones_array.shape}")


# 使用示例
if __name__ == "__main__":
    input_folder = "/home/hdd1/sunao/HumanML3D/test/"  # 替换为实际的输入文件夹路径
    output_folder = "/home/hdd1/sunao/HumanML3D/frame_label/"  # 替换为实际的输出文件夹路径

    create_ones_from_npy_files(input_folder, output_folder)
