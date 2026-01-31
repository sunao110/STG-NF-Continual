import json
import math
import os
import re
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
from utils.data_utils import normalize_pose
from utils.pose_utils import gen_clip_seg_data_np, get_ab_labels, gen_clip_seg_data_np_for_humanml3d, \
    gen_clip_seg_data_np_for_humanml3d_test
from torch.utils.data import DataLoader


class PoseSegDataset(Dataset):
    """
    Generates a dataset with two objects, a np array holding sliced pose sequences
    and an object array holding file name, person index and start time for each sliced seq


    If path_to_patches is provided uses pre-extracted patches. If lmdb_file or vid_dir are
    provided extracts patches from them, while hurting performance.
    """

    def __init__(self, path_dir, path_to_vid_dir=None, normalize_pose_segs=True, evaluate=False, mode='train',
                 cur_task=0, dataset_name=None,
                 **dataset_args):
        super().__init__()
        self.args = dataset_args
        self.data_numpy_path = path_dir
        self.normalize_pose_segs = normalize_pose_segs
        self.path_to_vid_dir = path_to_vid_dir
        self.eval = evaluate
        self.transform_list = dataset_args.get('trans_list', None)
        self.mode = mode
        if self.transform_list is None:
            self.apply_transforms = False
            self.num_transform = 1
        else:
            self.apply_transforms = True
            self.num_transform = len(self.transform_list)
        self.seg_len = dataset_args.get('seg_len', 12)
        self.seg_stride = dataset_args.get('seg_stride', 1)

        print('load {} dataset on {} for Task {}'.format(mode, dataset_name, cur_task))
        if mode == 'train':
            # 训练集: Task0 normal data Taski abnormal data
            if cur_task == 0:
                # 正常数据集
                self.segs_data_np = gen_dataset(self.data_numpy_path, **dataset_args)
                self.labels = np.ones(self.segs_data_np.shape[0])  # 1
            else:
                # 异常数据集
                self.segs_data_np = gen_dataset(self.data_numpy_path, **dataset_args)
                print("Num of abnormal sapmles: {}".format(self.segs_data_np.shape[0], ))
                self.labels = - np.ones(self.segs_data_np.shape[0])  # -1
        else:
            # 测试集 path_dir 是 dict  需要加载每帧的标签
            if not isinstance(self.data_numpy_path, dict):
                raise ValueError("test path_dir must be a dict")
            # path_dir[0]是normal path_dir[i]是abnormal
            total_segs_data_np = []
            total_labels = []
            segs_meta = []
            for i, path_dir in enumerate(self.data_numpy_path['data']):
                if path_dir == '/home/hdd1/sunao/HumanML3D/test':
                    segs_data_np, pose_segs_meta = gen_test_dataset(self.data_numpy_path['data'][0],
                                                               **dataset_args)
                    labels = np.ones(segs_data_np.shape[0])  # 1
                else:
                    segs_data_np, pose_segs_meta = gen_test_dataset(self.data_numpy_path['data'][i],
                                                               **dataset_args)
                    labels = -np.ones(segs_data_np.shape[0])  # -1

                total_segs_data_np.append(segs_data_np)
                total_labels.append(labels)
                segs_meta += pose_segs_meta

            self.segs_data_np = np.concatenate(total_segs_data_np, axis=0)
            self.labels = np.concatenate(total_labels, axis=0)
            self.segs_meta = segs_meta

        self.num_samples, self.C, self.T, self.V = self.segs_data_np.shape

    def __getitem__(self, index):
        # Select sample and augmentation. I.e. given 5 samples and 2 transformations,
        # sample 7 is data sample 7%5=2 and transform is 7//5=1
        if self.apply_transforms:
            sample_index = index % self.num_samples
            trans_index = math.floor(index / self.num_samples)
            data_numpy = np.array(self.segs_data_np[sample_index])
            data_transformed = self.transform_list[trans_index](data_numpy)
        else:
            sample_index = index
            data_transformed = np.array(self.segs_data_np[index])
            trans_index = 0  # No transformations

        if self.normalize_pose_segs:
            data_transformed = normalize_pose(data_transformed.transpose((1, 2, 0))[None, ...],
                                              **self.args).squeeze(axis=0).transpose(2, 0, 1)

        ret_arr = [data_transformed, trans_index]

        ret_arr += [self.labels[sample_index]]
        return ret_arr

    def get_all_data(self, normalize_pose_segs=True):
        if normalize_pose_segs:
            segs_data_np = normalize_pose(self.segs_data_np.transpose((0, 2, 3, 1)), **self.args).transpose(
                (0, 3, 1, 2))
        else:
            segs_data_np = self.segs_data_np
        if self.num_transform == 1 or self.eval:
            return list(segs_data_np)
        return segs_data_np

    def __len__(self):
        return self.num_transform * self.num_samples


def get_dataset_and_loader_for_incremental_task(args, trans_list, cur_task, dataset_name, only_test=False):
    loader_args = {'batch_size': args.batch_size, 'num_workers': args.num_workers, 'pin_memory': True}
    dataset_args = {'scale': args.norm_scale, 'scale_proportional': args.prop_norm_scale,
                    'seg_len': args.seg_len, "dataset": dataset_name}
    dataset, loader = dict(), dict()

    print("load dataset: {} ...".format(dataset_name))

    for split in args.pose_path.keys():
        evaluate = split == 'test'
        normalize_pose_segs = args.global_pose_segs
        dataset_args['trans_list'] = trans_list[:args.num_transform] if split == 'train' else None
        dataset_args['seg_stride'] = args.seg_stride if split == 'train' else 1  # No strides for test set

        dataset[split] = PoseSegDataset(args.pose_path[split],
                                        normalize_pose_segs=normalize_pose_segs,
                                        evaluate=evaluate,
                                        mode=split,
                                        cur_task=cur_task,
                                        dataset_name=dataset_name,
                                        **dataset_args)
        loader[split] = DataLoader(dataset[split], **loader_args, shuffle=(split == 'train'))

    if only_test:
        loader['train'] = None
    return dataset, loader


def gen_dataset(root_dir, **dataset_args):
    segs_data_np = []
    # global_data = []
    start_ofst = dataset_args.get('start_ofst', 0)
    seg_stride = dataset_args.get('seg_stride', 1)
    seg_len = dataset_args.get('seg_len', 24)

    npy_file_list = os.listdir(root_dir)
    npy_file__name_list = sorted([fn for fn in npy_file_list if fn.endswith('.npy')])
    for npy_name in tqdm(npy_file__name_list):
        data_npy = None

        if root_dir:
            npy_file_path = os.path.join(root_dir, npy_name)
            data_npy = np.load(npy_file_path)

        # 有些异常的文件
        if len(data_npy.shape) != 3:
            continue
        # 裁剪成多个样本
        data_npy = gen_clip_seg_data_np_for_humanml3d(data_npy, start_ofst, seg_stride, seg_len, )

        segs_data_np.append(data_npy)

    segs_data_np = np.concatenate(segs_data_np, axis=0)

    segs_data_np = np.transpose(segs_data_np, (0, 3, 1, 2)).astype(np.float32)

    return segs_data_np


# 测试集需要加载每帧的标签
def gen_test_dataset(root_dir, **dataset_args):
    segs_data_np = []
    segs_meta = []
    start_ofst = dataset_args.get('start_ofst', 0)
    seg_stride = dataset_args.get('seg_stride', 1)
    seg_len = dataset_args.get('seg_len', 24)

    npy_file_list = os.listdir(root_dir)
    npy_file__name_list = sorted([fn for fn in npy_file_list if fn.endswith('.npy')])
    for npy_name in tqdm(npy_file__name_list):
        data_npy = None

        if root_dir:
            npy_file_path = os.path.join(root_dir, npy_name)
            data_npy = np.load(npy_file_path)

        # 有些异常的文件
        if len(data_npy.shape) != 3:
            continue
        # 裁剪成多个样本
        data_npy, pose_segs_meta = gen_clip_seg_data_np_for_humanml3d_test(data_npy, start_ofst,
                                                                           seg_stride, seg_len, int(
                os.path.splitext(os.path.basename(npy_name))[0]))

        segs_data_np.append(data_npy)
        segs_meta += pose_segs_meta

    segs_data_np = np.concatenate(segs_data_np, axis=0)
    segs_data_np = np.transpose(segs_data_np, (0, 3, 1, 2)).astype(np.float32)

    return segs_data_np, segs_meta
