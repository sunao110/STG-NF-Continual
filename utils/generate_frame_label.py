import numpy as np
import os

base_path = '/home/hdd1/sunao/ACMDM/frame_label/'


def generate_npy_with_zeros(length, zero_indices, save_path):
    """
    生成指定长度的一维 numpy 数组，在指定索引位置为 0，其他位置为 1

    Args:
        length: 数组长度
        zero_indices: 需要设置为 0 的索引列表
        save_path: 保存路径
    """
    # 创建全1数组
    arr = np.ones(length, dtype=np.int32)

    # 将指定索引位置设置为0
    for idx in zero_indices:
        if 0 <= idx < length:  # 检查索引是否在有效范围内
            arr[idx] = 0
        else:
            print(f"警告: 索引 {idx} 超出数组范围 [0, {length - 1}]")

    # 确保保存目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 保存数组
    np.save(save_path, arr)

    print(f"数组已保存至: {save_path}")
    print(f"数组长度: {length}")
    print(f"零值索引: {zero_indices}")
    print(f"数组形状: {arr.shape}")


# 使用示例
if __name__ == "__main__":
    # 示例参数
    length = 196  # 数组长度

    tuples_list = [
        (5, 93),
        (6, 179),
        (13, 160),
        (15, 128),
        (29, 55),
        (32, 129),
        (35, 175),
        (37, 120),
        (51, 191),
        (54, 76),
        (57, 56),
        (58, 73),
        (70, 119),
        (71, 69),
        (78, 109),
        (86, 116),
        (87, 113),
        (88, 77),
        (101, 140),
        (103, 72),
        (107, 128),
        (110, 65),
        (115, 148),
        (119, 53),
        (120, 45),
        (123, 45),
        (125, 73),
        (129, 76),
        (134, 172),
        (136, 92),
        (140, 173),
        (142, 124),
        (145, 176),
        (147, 121),
        (151, 49),
        (172, 156),
        (182, 144),
        (184, 164),
        (206, 128),
        (212, 153),
        (213, 165),
        (233, 145),
        (256, 161),
        (258, 148),
        (262, 65),
        (267, 165),
        (273, 97),
        (285, 65),
        (288, 148),
        (290, 109),
        (293, 177),
        (324, 57),
        (339, 77),
        (350, 81),
        (386, 157),
        (394, 77),
        (398, 101),
        (399, 137),
        (401, 153),
        (415, 57),
        (424, 45),
        (428, 61),
        (430, 89),
        (435, 141),
        (455, 111),
        (460, 89),
    ]

    for i in range(len(tuples_list)):
        zero_indices = [j for j in range(tuples_list[i][1], length)]
        save_path = os.path.join(base_path, f"{tuples_list[i][0]}.npy")
        generate_npy_with_zeros(length, zero_indices, save_path)
