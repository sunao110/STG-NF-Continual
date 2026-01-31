import os
import argparse


def delete_npy_files(directory, num_files):
    # 获取目录下所有的 .npy 文件
    npy_files = [f for f in os.listdir(directory) if f.endswith('.npy')]

    # 如果文件数量少于要删除的数量，则全部删除
    if len(npy_files) < num_files:
        print(f"目录中只有 {len(npy_files)} 个 .npy 文件，将全部删除。")
        num_files = len(npy_files)

    # 删除指定数量的 .npy 文件
    for i in range(num_files):
        file_path = os.path.join(directory, npy_files[i])
        os.remove(file_path)
        print(f"已删除文件: {file_path}")


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="删除指定目录下的指定个数的 .npy 文件")
    # parser.add_argument("directory", type=str, default='/home/hdd1/sunao/ACMDM/test', help="要删除文件的目录路径")
    # parser.add_argument("num_files", type=int, default=349, help="要删除的 .npy 文件数量")

    # args = parser.parse_args()

    delete_npy_files('/home/hdd1/sunao/ACMDM/test', 349)
