import numpy as np
import os
import glob

def modify_last_element_in_npy_files(directory_path, save_back=True):
    """
    将目录下所有npy文件的最后一个元素改为0

    Args:
        directory_path: 包含npy文件的目录路径
        save_back: 是否覆盖原文件，默认为True
    """
    # 获取目录下所有 .npy 文件
    npy_files = glob.glob(os.path.join(directory_path, "*.npy"))

    print(f"找到 {len(npy_files)} 个npy文件")

    for i, file_path in enumerate(npy_files):
        try:
            # 加载npy文件
            data = np.load(file_path)

            # 检查数组是否为空
            if data.size == 0:
                print(f"警告: 文件 {file_path} 为空数组，跳过")
                continue

            # 修改最后一个元素为0
            original_last_value = data.flat[-1]
            data.flat[-1] = 0

            # 保存文件
            if save_back:
                np.save(file_path, data)
                print(f"[{i+1}/{len(npy_files)}] 已修改并保存: {os.path.basename(file_path)}, "
                      f"原最后元素: {original_last_value}, 新最后元素: 0")

        except Exception as e:
            print(f"错误: 处理文件 {file_path} 时出错 - {e}")

    print("批量修改完成！")

if __name__ == "__main__":

    directory_path = '/home/hdd1/sunao/ACMDM/frame_label'


    modify_last_element_in_npy_files(directory_path)
