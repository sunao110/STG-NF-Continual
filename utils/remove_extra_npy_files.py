import os

def remove_extra_npy_files(dir1, dir2):
    # 获取目录1中的所有.npy文件名（不包括路径）
    dir1_npy_files = {f for f in os.listdir(dir1) if f.endswith('.npy')}

    # 遍历目录2中的所有.npy文件
    for filename in os.listdir(dir2):
        if filename.endswith('.npy') and filename not in dir1_npy_files:
            file_path = os.path.join(dir2, filename)
            print(f"Deleting: {file_path}")
            os.remove(file_path)

if __name__ == "__main__":
    # 示例用法
    dir1 = '/home/hdd1/sunao/ACMDM/frame_label'  # 替换为目录1的路径
    dir2 = '/home/hdd1/sunao/ACMDM/test'  # 替换为目录2的路径

    remove_extra_npy_files(dir1, dir2)
