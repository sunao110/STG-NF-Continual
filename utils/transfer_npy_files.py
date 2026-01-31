import os
import shutil
import random
import argparse
from pathlib import Path


def transfer_npy_files(source_dir, target_dir, transfer_ratio=0.2, random_seed=42):
    """
    将source_dir下的npy文件按指定比例转移到target_dir

    参数：
    source_dir: 源文件夹路径（存放所有npy文件）
    target_dir: 目标文件夹路径
    transfer_ratio: 转移文件占比（0-1之间，默认0.2）
    random_seed: 随机种子（保证划分结果可复现）
    """
    # 1. 校验输入参数
    source_path = Path(source_dir)
    target_path = Path(target_dir)

    # 检查源文件夹是否存在
    if not source_path.exists():
        raise FileNotFoundError(f"源文件夹不存在：{source_dir}")

    # 创建目标文件夹（不存在则创建）
    target_path.mkdir(parents=True, exist_ok=True)

    # 2. 获取所有npy文件
    npy_files = [f for f in source_path.glob("*.npy") if f.is_file()]
    if len(npy_files) == 0:
        raise ValueError(f"源文件夹 {source_dir} 中未找到任何.npy文件")

    # 3. 随机打乱并确定转移文件
    random.seed(random_seed)  # 固定随机种子，结果可复现
    random.shuffle(npy_files)

    transfer_num = int(len(npy_files) * transfer_ratio)
    transfer_files = npy_files[:transfer_num]
    remaining_files = npy_files[transfer_num:]

    # 4. 打印转移信息
    print(f"===== 转移信息 =====")
    print(f"总.npy文件数：{len(npy_files)}")
    print(f"转移文件数：{len(transfer_files)} (占比 {transfer_ratio:.1%})")
    print(f"剩余文件数：{len(remaining_files)} (占比 {1 - transfer_ratio:.1%})")
    print(f"源文件夹路径：{source_dir}")
    print(f"目标文件夹路径：{target_dir}")
    confirm = input("\n确认执行文件转移？[y/n]：")
    if confirm.lower() != 'y':
        print("操作已取消")
        return

    # 5. 转移文件
    print("\n===== 开始转移文件 =====")
    for idx, file in enumerate(transfer_files, 1):
        dest_path = target_path / file.name
        shutil.move(str(file), str(dest_path))  # move是剪切，copy2是复制（保留元数据）
        print(f"[{idx}/{len(transfer_files)}] 转移：{file.name} → {dest_path}")

    print("\n===== 转移完成 =====")
    print(f"成功转移：{len(transfer_files)} 个文件")
    print(f"源文件夹剩余：{len(remaining_files)} 个文件")


def main():
    # 命令行参数配置
    parser = argparse.ArgumentParser(description='转移指定百分比的.npy文件')
    parser.add_argument('--source', '-s', required=True, help='源文件夹路径（存放所有npy文件）')
    parser.add_argument('--target', '-t', required=True, help='目标文件夹路径')
    parser.add_argument('--ratio', '-r', type=float, default=0.5, help='转移文件占比（默认0.2，即20%）')
    parser.add_argument('--seed', '-sd', type=int, default=42, help='随机种子（默认42）')

    args = parser.parse_args()

    # 执行转移
    try:
        transfer_npy_files(
            source_dir=args.source,
            target_dir=args.target,
            transfer_ratio=args.ratio,
            random_seed=args.seed
        )
    except Exception as e:
        print(f"执行出错：{e}")


if __name__ == '__main__':
    main()
