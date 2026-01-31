import os
import random
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from models.STG_NF.model_pose import STG_NF
from models.training import Trainer
from utils.data_utils import trans_list
from utils.optim_init import init_optimizer, init_scheduler
from args import create_exp_dirs
from args import init_parser, init_sub_args
from dataset_humanml3d import get_dataset_and_loader_for_incremental_task
from utils.train_utils import dump_args, init_model_params
from utils.scoring_utils import score_dataset
from utils.train_utils import calc_num_of_params


def main():
    parser = init_parser()
    args = parser.parse_args()

    if args.seed == 999:  # Record and init seed
        args.seed = torch.initial_seed()
        np.random.seed(0)
    else:
        random.seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        torch.manual_seed(args.seed)
        np.random.seed(0)

    args.ckpt_dir = create_exp_dirs(args.exp_dir)
    # dataset_list = ['HumanML3D']
    dataset_list = ['HumanML3D', 'ACMDM']

    model_args = init_model_params(args)
    model = STG_NF(**model_args)
    start_task = 0

    # 加载预训练模型
    if args.load_pretrained_checkpoint:
        print("load pretrained checkpoint ======> {}".format(args.load_pretrained_checkpoint))
        checkpoint = torch.load(args.load_pretrained_checkpoint)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        model.set_actnorm_init()
        start_task = 1

    for cur_task in range(start_task,len(dataset_list)):
        print("============================Task: {}============================".format(cur_task))
        epochs = args.epochs if cur_task == 0 else args.incremental_epochs
        # 上一个任务的断点
        pretrained = os.path.join(args.ckpt_dir, 'task_' + str(cur_task - 1),
                                  'epoch_' + str(args.epochs) + '_checkpoint.pth.tar') if cur_task > 1 else None
        args, model_args = init_sub_args(args, dataset=dataset_list[cur_task])

        print(args)
        dataset, loader = get_dataset_and_loader_for_incremental_task(args, cur_task=cur_task,
                                                                      dataset_name=dataset_list[cur_task],
                                                                      trans_list=trans_list, )

        if 'train' in dataset:
            print(
                "# train sample shape: {}, label: {}, num: {} \n".format(
                    dataset['train'][0][0].shape, dataset['train'][0][-1], len(dataset['train'])))

        if 'test' in dataset:
            print(
                "# test sample shape: {}, label: {}, num: {}\n".format(
                    dataset['test'][0][0].shape, dataset['test'][0][-1], len(dataset['test'])))

        trainer = Trainer(args, model, loader['train'] if 'train' in dataset else None, loader['test'] if 'test' in loader else None,
                          optimizer_f=init_optimizer(args.model_optimizer, lr=args.model_lr),
                          scheduler_f=init_scheduler(args.model_sched, lr=args.model_lr, epochs=epochs))

        if pretrained:
            print("load checkpoint ======> {}".format(pretrained))
            trainer.load_checkpoint(pretrained)


        if 'train' in dataset:
            writer = SummaryWriter()
            trainer.train(log_writer=writer, task=cur_task, num_epochs=epochs)
            dump_args(args, args.ckpt_dir)

        if 'test' in dataset:
            normality_scores = trainer.test()
            auc, scores = score_dataset(normality_scores, dataset["test"].segs_meta, args=args, only_test=True)

            # Logging and recording results
            print("\n-------------------------------------------------------")
            print("\033[92m Done with {}% AuC for {} samples\033[0m".format(auc * 100, scores.shape[0]))
            print("-------------------------------------------------------\n\n")


if __name__ == '__main__':
    main()
