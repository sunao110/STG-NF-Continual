import os

def rename_npy_files(input_folder):
    """
    将文件夹下的所有.npy文件按递增数字重命名

    Args:
        input_folder (str): 输入文件夹路径
    """
    # 获取所有.npy文件
    npy_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.npy')]

    # 按当前名称排序，保证重命名顺序一致
    npy_files.sort()

    # 重命名文件
    for idx, old_filename in enumerate(npy_files, start=1):
        old_path = os.path.join(input_folder, old_filename)

        # 生成新的文件名（保持.npy扩展名）
        new_filename = f"{idx}.npy"
        new_path = os.path.join(input_folder, new_filename)

        # 重命名文件
        os.rename(old_path, new_path)
        print(f"重命名: {old_filename} -> {new_filename}")


# 使用示例
if __name__ == "__main__":
    folder_path = "/home/hdd1/sunao/ACMDM/test"  # 替换为实际路径
    rename_npy_files(folder_path)
