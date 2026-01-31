import os
import shutil


def move_npy_files_from_list(txt_file_path, source_folder, target_folder):
    """
    从txt文件中加载文件名，移动源文件夹中对应的npy文件到目标文件夹

    Args:
        txt_file_path: 包含文件名的txt文件路径
        source_folder: 源文件夹路径
        target_folder: 目标文件夹路径
    """
    # 确保目标文件夹存在
    os.makedirs(target_folder, exist_ok=True)

    # 读取txt文件中的文件名
    with open(txt_file_path, 'r', encoding='utf-8') as f:
        file_names = [line.strip() for line in f if line.strip()]

    # 移动文件
    moved_count = 0
    for file_name in file_names:
        # 确保文件名以.npy结尾
        if not file_name.lower().endswith('.npy'):
            file_name += '.npy'

        source_path = os.path.join(source_folder, file_name)
        target_path = os.path.join(target_folder, file_name)

        # 检查源文件是否存在
        if os.path.exists(source_path):
            shutil.move(source_path, target_path)
            print(f"已移动: {file_name}")
            moved_count += 1
        else:
            print(f"文件不存在: {source_path}")

    print(f"完成移动操作，共移动了 {moved_count} 个文件")


# 使用示例
if __name__ == "__main__":
    txt_file_path = "/home/hdd1/sunao/ACMDM/task1_train_sample_order.txt"  # 包含文件名的txt文件路径
    source_folder = "/home/hdd1/sunao/ACMDM/test/"  # 源文件夹路径
    target_folder = "/home/hdd1/sunao/ACMDM/train/"  # 目标文件夹路径

    move_npy_files_from_list(txt_file_path, source_folder, target_folder)
