import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, FFMpegFileWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d.axes3d as p3
# import cv2
from textwrap import wrap
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage

kinematic_tree_humanml3d = [
    [0, 3, 6, 9, 12, 15],  # 躯干-头部链
    [0, 2, 5, 8, 11],  # 左下肢链：骨盆→左髋→左大腿→左小腿→左脚
    [0, 1, 4, 7, 10],  # 右下肢链：骨盆→右髋→右大腿→右小腿→右脚
    [9, 14, 17, 19, 21],  # 左上肢链：骨盆→腹部→胸部→左肩→左前臂→左手
    [9, 13, 16, 18, 20]  # 右上肢链：骨盆→腹部→胸部→右肩→右大臂→右前臂→右手→右手补充
]

kinematic_tree_acmdm = [
    [0, 3, 6, 9, 12, 15],  # 躯干-头部链
    [0, 2, 5, 8, 11],  # 左下肢链：骨盆→左髋→左大腿→左小腿→左脚
    [0, 1, 4, 7, 10],  # 右下肢链：骨盆→右髋→右大腿→右小腿→右脚
    [9, 14, 17, 19, 21],  # 左上肢链：骨盆→腹部→胸部→左肩→左前臂→左手
    [9, 13, 16, 18, 20]  # 右上肢链：骨盆→腹部→胸部→右肩→右大臂→右前臂→右手→右手补充
]


def plot_3d_motion(save_path, joints, kinematic_tree, title, scores, figsize=(3, 3), fps=120, radius=3,
                   vis_mode='default', gt_frames=[]):
    matplotlib.use('Agg')

    title_per_frame = type(title) == list
    if title_per_frame:
        assert len(title) == len(joints), 'Title length should match the number of frames'
        title = ['\n'.join(wrap(s, 20)) for s in title]
    else:
        title = '\n'.join(wrap(title, 20))

    def init():
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([0, radius])
        ax.set_zlim3d([-radius / 3., radius * 2 / 3.])
        ax.grid(b=False)

    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        ## Plot a plane XZ
        verts = [
            [minx, miny, minz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, miny, minz]
        ]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)

    #         return ax

    # (seq_len, joints_num, 3)
    data = joints.copy().reshape(len(joints), -1, 3)

    fig = plt.figure(figsize=figsize)
    plt.tight_layout()
    ax = p3.Axes3D(fig)
    init()
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    colors_blue = ["#4D84AA", "#5B9965", "#61CEB9", "#34C1E2", "#80B79A"]  # GT color
    colors_orange = ["#DD5A37", "#D69E00", "#B75A39", "#FF6D00", "#DDB50E"]  # Generation color
    colors = colors_orange
    if vis_mode == 'upper_body':  # lower body taken fixed to input motion
        colors[0] = colors_blue[0]
        colors[1] = colors_blue[1]
    elif vis_mode == 'gt':
        colors = colors_blue

    n_frames = data.shape[0]
    #     print(dataset.shape)

    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    trajec = data[:, 0, [0, 2]]  # memorize original x,z pelvis values

    # locate x,z pelvis values of ** each frame ** at zero
    data[..., 0] -= data[:, 0:1, 0]
    data[..., 2] -= data[:, 0:1, 2]

    #     print(trajec.shape)

    def update(index):
        # sometimes index is equal to n_frames/fps due to floating point issues. in such case, we duplicate the last frame
        index = min(n_frames - 1, int(index * fps))
        ax.clear()
        ax.view_init(elev=120, azim=-90)
        ax.dist = 7.5

        # Dynamic title
        if title_per_frame:
            _title = title[index]
        else:
            _title = title
        # 保留4位小数
        rounded_number = round(scores[index], 4)
        _title += f' [{index}]' + f'score: {rounded_number}'
        fig.suptitle(_title, fontsize=6)

        plot_xzPlane(MINS[0] - trajec[index, 0], MAXS[0] - trajec[index, 0], 0, MINS[2] - trajec[index, 1],
                     MAXS[2] - trajec[index, 1])

        used_colors = colors_blue if index in gt_frames else colors
        for i, (chain, color) in enumerate(zip(kinematic_tree, used_colors)):
            if i < 5:
                linewidth = 4.0
            else:
                linewidth = 2.0
            ax.plot3D(data[index, chain, 0], data[index, chain, 1], data[index, chain, 2], linewidth=linewidth,
                      color=color)
        #         print(trajec[:index, 0].shape)

        plt.axis('off')
        ax.set_axis_off()
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

        # Hide grid lines
        ax.grid(False)

        # Hide axes ticks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        return mplfig_to_npimage(fig)

    print(n_frames / fps)

    ani = VideoClip(update, duration=n_frames / fps)

    ani.write_videofile(save_path, fps=fps, codec='libx264')

    plt.close()
    return ani


if __name__ == "__main__":
    import os
    # 加载骨骼序列npy文件
    skeleton_file_path = '/home/hdd1/sunao/ACMDM/test'
    # 加载分数npy文件
    scores_file_path = '/home/hdd1/sunao/ACMDM/scores'

    npy_files = [f for f in os.listdir(skeleton_file_path) if f.lower().endswith('.npy')]
    # 排序保证测试集文件加载的顺序相同
    clip_list = sorted(fn for fn in npy_files if fn.endswith('.npy'))

    for filename in clip_list:
        # 加载骨骼数据
        joints_data = np.load(os.path.join(skeleton_file_path, filename))
        print(f"加载的骨骼数据形状: {joints_data.shape}")
        scores_data = np.load(os.path.join(scores_file_path, filename))
        print(f"加载的骨骼数据形状: {scores_data.shape}")

        # 归一化得分
        scores_data = (scores_data - np.min(scores_data)) / (np.max(scores_data) - np.min(scores_data))
        scores_data[:12] = -1
        scores_data[-12:] = -1
        # 生成动画
        print("正在生成骨骼动画...")
        animation_clip = plot_3d_motion(
            save_path=f"/home/hdd1/sunao/ACMDM/video/{filename}.mp4",
            joints=joints_data,
            kinematic_tree=kinematic_tree_humanml3d,
            title="Skeleton Motion Visualization",
            fps=1,
            scores=scores_data,
        )

        print(f"骨骼动画已保存: {filename}.mp4")
