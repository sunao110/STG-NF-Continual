import os
import shutil
import random
import argparse
from pathlib import Path


def split_npy_files(source_dir, train_dir, val_dir, train_ratio=0.8, random_seed=42):
    """
    将source_dir下的npy文件按比例划分到train_dir和val_dir

    参数：
    source_dir: 源文件夹路径（存放所有npy文件）
    train_dir: 训练集目标文件夹路径
    val_dir: 测试集目标文件夹路径
    train_ratio: 训练集占比（0-1之间，默认0.8）
    random_seed: 随机种子（保证划分结果可复现）
    """
    # 1. 校验输入参数
    source_path = Path(source_dir)
    train_path = Path(train_dir)
    val_path = Path(val_dir)

    # 检查源文件夹是否存在
    if not source_path.exists():
        raise FileNotFoundError(f"源文件夹不存在：{source_dir}")

    # 创建目标文件夹（不存在则创建）
    train_path.mkdir(parents=True, exist_ok=True)
    val_path.mkdir(parents=True, exist_ok=True)

    # 2. 获取所有npy文件
    npy_files = [f for f in source_path.glob("*.npy") if f.is_file()]
    if len(npy_files) == 0:
        raise ValueError(f"源文件夹 {source_dir} 中未找到任何.npy文件")

    # 3. 随机打乱并划分文件
    random.seed(random_seed)  # 固定随机种子，结果可复现
    random.shuffle(npy_files)

    train_num = int(len(npy_files) * train_ratio)
    train_files = npy_files[:train_num]
    val_files = npy_files[train_num:]

    # 4. 打印划分信息（确认）
    print(f"===== 划分信息 =====")
    print(f"总.npy文件数：{len(npy_files)}")
    print(f"训练集文件数：{len(train_files)} (占比 {train_ratio:.1%})")
    print(f"验证集文件数：{len(val_files)} (占比 {1 - train_ratio:.1%})")
    print(f"训练集保存路径：{train_dir}")
    print(f"验证集保存路径：{val_dir}")
    confirm = input("\n确认执行文件转移？[y/n]：")
    if confirm.lower() != 'y':
        print("操作已取消")
        return

    # 5. 转移文件
    print("\n===== 开始转移文件 =====")
    # 转移训练集
    for idx, file in enumerate(train_files, 1):
        dest_path = train_path / file.name
        shutil.move(str(file), str(dest_path))  # move是剪切，copy2是复制（保留元数据）
        print(f"[{idx}/{len(train_files)}] 训练集：{file.name} → {dest_path}")

    # 转移验证集
    for idx, file in enumerate(val_files, 1):
        dest_path = val_path / file.name
        shutil.move(str(file), str(dest_path))
        print(f"[{idx}/{len(val_files)}] 验证集：{file.name} → {dest_path}")

    print("\n===== 转移完成 =====")
    print(f"训练集：{len(train_files)} 个文件")
    print(f"验证集：{len(val_files)} 个文件")


def main():
    # 命令行参数配置
    parser = argparse.ArgumentParser(description='分割数据集')
    parser.add_argument('--source', '-s', required=True, help='源文件夹路径（存放所有npy文件）')
    parser.add_argument('--train', '-t', required=True, help='训练集目标文件夹路径')
    parser.add_argument('--test', '-v', required=True, help='验证集目标文件夹路径')
    parser.add_argument('--ratio', '-r', type=float, default=0.8, help='训练集占比（默认0.8）')
    parser.add_argument('--seed', '-sd', type=int, default=999, help='随机种子（默认999）')

    args = parser.parse_args()

    # 执行划分
    try:
        split_npy_files(
            source_dir=args.source,
            train_dir=args.train,
            val_dir=args.test,
            train_ratio=args.ratio,
            random_seed=args.seed
        )
    except Exception as e:
        print(f"执行出错：{e}")


if __name__ == '__main__':
    main()