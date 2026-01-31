import os
import re
import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from dataset import shanghaitech_hr_skip
from utils.draw_utils import plot_score_curve


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def score_dataset(score, metadata, args=None, only_test=False):
    gt_arr, scores_arr = get_dataset_scores(score, metadata, args=args, only_test=only_test)

    if not only_test:
        scores_arr = smooth_scores(scores_arr)
        gt_np = np.concatenate(gt_arr)
        scores_np = np.concatenate(scores_arr)
        auc = score_auc(scores_np, gt_np)
        return auc, scores_np

    return None, None



def get_dataset_scores(scores, metadata, args=None, only_test=False):
    dataset_gt_arr = []
    dataset_scores_arr = []
    metadata_np = np.array(metadata)

    per_frame_scores_root = args.pose_path['test']['frame_label'][0] if not only_test else None
    npy_files = [f for f in os.listdir(args.pose_path['test']['data'][0]) if f.lower().endswith('.npy')]
    # 排序保证测试集文件加载的顺序相同
    clip_list = sorted(fn for fn in npy_files if fn.endswith('.npy'))

    print("Scoring {} clips".format(len(clip_list)))
    for clip in tqdm(clip_list):
        clip_gt, clip_score = get_clip_score(scores, clip, metadata_np, metadata, per_frame_scores_root, args, 196)
        if clip_gt is not None:
            dataset_gt_arr.append(clip_gt)
        if clip_score is not None:
            dataset_scores_arr.append(clip_score)

    scores_np = np.concatenate(dataset_scores_arr, axis=0)
    # 替换无穷值为当前tensor元素的最值，防止后续？AUC计算出错
    scores_np[scores_np == np.inf] = scores_np[scores_np != np.inf].max()
    scores_np[scores_np == -1 * np.inf] = scores_np[scores_np != -1 * np.inf].min()
    index = 0
    for score in range(len(dataset_scores_arr)):
        for t in range(dataset_scores_arr[score].shape[0]):
            dataset_scores_arr[score][t] = scores_np[index]
            index += 1

    return dataset_gt_arr, dataset_scores_arr


def score_auc(scores_np, gt):
    scores_np[scores_np == np.inf] = scores_np[scores_np != np.inf].max()
    scores_np[scores_np == -1 * np.inf] = scores_np[scores_np != -1 * np.inf].min()
    auc = roc_auc_score(gt, scores_np)
    return auc


def smooth_scores(scores_arr, sigma=7):
    for s in range(len(scores_arr)):
        for sig in range(1, sigma):
            scores_arr[s] = gaussian_filter1d(scores_arr[s], sigma=sig)
    return scores_arr

# 取同一场景下的最低分数person作为该样本的score
def get_clip_score(scores, clip, metadata_np, metadata, per_frame_scores_root, args, n_frames):
    order = int(os.path.splitext(os.path.basename(clip))[0])
    # 归属同一文件的样本帧序号，用于拼接
    score_inds = np.where(metadata_np[:, 0] == order)[0]
    pid_scores = scores[score_inds]
    # 标签
    if per_frame_scores_root is not None:
        clip_res_fn = os.path.join(per_frame_scores_root, clip)
        clip_gt = np.load(clip_res_fn)
        clip_score = np.ones(clip_gt.shape[0]) * np.inf

        pid_frame_inds = np.array([metadata[i][1] for i in score_inds]).astype(int)
        clip_score[pid_frame_inds + int(args.seg_len / 2)] = pid_scores
        return clip_gt, clip_score

    # 绘制图像
    plot_score_curve(pid_scores, f"score_curve_{order}.png", offset=int(args.seg_len / 2))
    # 保存每一帧的得分
    if args.pose_path['test']['scores_path']:
        save_scores = np.zeros(n_frames)
        frame_inds = np.arange(pid_scores.shape[0]).astype(int)
        save_scores[frame_inds + int(args.seg_len / 2)] = pid_scores
        np.save(os.path.join(args.pose_path['test']['scores_path'], f"{order}.npy"), save_scores)

    return None, None
