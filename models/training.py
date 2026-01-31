"""
Train\Test helper, based on awesome previous work by https://github.com/amirmk89/gepc
"""

import os
import time
import shutil
import torch
import torch.optim as optim
from tqdm import tqdm

from utils.scoring_utils import smooth_scores, score_auc


def adjust_lr(optimizer, epoch, lr=None, lr_decay=None, scheduler=None):
    if scheduler is not None:
        scheduler.step()
        new_lr = scheduler.get_lr()[0]
    elif (lr is not None) and (lr_decay is not None):
        new_lr = lr * (lr_decay ** epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
    else:
        raise ValueError('Missing parameters for LR adjustment')
    return new_lr


def compute_loss(nll, reduction="mean", mean=0):
    if reduction == "mean":
        losses = {"nll": torch.mean(nll)}
    elif reduction == "logsumexp":
        losses = {"nll": torch.logsumexp(nll, dim=0)}
    elif reduction == "exp":
        losses = {"nll": torch.exp(torch.mean(nll) - mean)}
    elif reduction == "none":
        losses = {"nll": nll}

    losses["total_loss"] = losses["nll"]

    return losses


class Trainer:
    def __init__(self, args, model, train_loader, test_loader,
                 optimizer_f=None, scheduler_f=None):
        self.model = model
        self.args = args
        self.train_loader = train_loader
        self.test_loader = test_loader
        # Loss, Optimizer and Scheduler
        if optimizer_f is None:
            self.optimizer = self.get_optimizer()
        else:
            self.optimizer = optimizer_f(self.model.parameters())
        if scheduler_f is None:
            self.scheduler = None
        else:
            self.scheduler = scheduler_f(self.optimizer)

    def get_optimizer(self):
        if self.args.optimizer == 'adam':
            if self.args.lr:
                return optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
            else:
                return optim.Adam(self.model.parameters())
        elif self.args.optimizer == 'adamx':
            if self.args.lr:
                return optim.Adamax(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
            else:
                return optim.Adamax(self.model.parameters())
        return optim.SGD(self.model.parameters(), lr=self.args.lr)

    def adjust_lr(self, epoch):
        return adjust_lr(self.optimizer, epoch, self.args.model_lr, self.args.model_lr_decay, self.scheduler)

    def optimizer_to_cuda(self, rank):
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda(rank)

    def save_checkpoint(self, epoch, is_best=False, filename=None, task=None):
        """
        state: {'epoch': cur_epoch + 1, 'state_dict': self.model.state_dict(),
                            'optimizer': self.optimizer.state_dict()}
        """
        state = self.gen_checkpoint_state(epoch)
        if filename is None:
            filename = 'checkpoint.pth.tar'

        state['args'] = self.args
        # 增量任务断点目录
        task_dir = os.path.join(self.args.ckpt_dir, 'task_' + str(task))
        if not os.path.exists(task_dir):
            os.makedirs(task_dir)

        path_join = os.path.join(task_dir, filename)
        torch.save(state, path_join)

        if is_best:
            shutil.copy(path_join, os.path.join(task_dir, 'checkpoint_best.pth.tar'))

    def load_checkpoint(self, filename):
        filename = filename
        try:
            checkpoint = torch.load(filename)
            # self.start_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'], strict=False)
            self.model.set_actnorm_init()
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("Checkpoint loaded successfully from '{}' at (epoch {})\n"
                  .format(filename, checkpoint['epoch']))
        except FileNotFoundError:
            print("No checkpoint exists from '{}'. Skipping...\n".format(self.args.ckpt_dir))

    def train(self, log_writer=None, clip=100, task=None, num_epochs=1):
        checkpoint_filename = '_checkpoint.pth.tar'
        start_epoch = 1
        self.model.train()
        self.model = self.model.to(self.args.device)
        self.optimizer_to_cuda(self.args.device)
        key_break = False

        for epoch in range(start_epoch, num_epochs + 1):
            if key_break:
                break

            print("Starting Epoch {} / {}".format(epoch, num_epochs))
            pbar = tqdm(self.train_loader)
            for itern, data_arr in enumerate(pbar):
                try:
                    data = [data.to(self.args.device, non_blocking=True) for data in data_arr]
                    label = data[-1]
                    samp = data[0]

                    z, nll = self.model(samp.float(), label=label)
                    if nll is None:
                        continue
                    losses = compute_loss(nll, reduction="mean")["total_loss"]
                    losses.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    pbar.set_description(f"Loss: {losses.item():.4f}")
                    log_writer.add_scalar('NLL Loss', losses.item(), epoch * len(self.train_loader) + itern)

                except KeyboardInterrupt:
                    print('Keyboard Interrupted. Save results? [yes/no]')
                    choice = input().lower()
                    if choice == "yes":
                        key_break = True
                        break
                    else:
                        exit(1)

            if (epoch % self.args.save_freq == 0) or (epoch == num_epochs):
                self.save_checkpoint(epoch, filename='epoch_' + str(epoch) + checkpoint_filename, task=task)
                print('Checkpoint Saved.')

            new_lr = self.adjust_lr(epoch)
            print('New LR: {0:.3e}'.format(new_lr))

    def test(self):
        self.model.eval()
        self.model.to(self.args.device)
        pbar = tqdm(self.test_loader)
        probs = torch.empty(0).to(self.args.device)
        print("Starting Test Eval")
        for itern, data_arr in enumerate(pbar):
            data = [data.to(self.args.device, non_blocking=True) for data in data_arr]
            samp = data[0]
            with torch.no_grad():
                z, nll = self.model(samp.float())
            probs = torch.cat((probs, -1 * nll), dim=0)
        prob_mat_np = probs.cpu().detach().numpy().squeeze().copy(order='C')
        return prob_mat_np

    def gen_checkpoint_state(self, epoch):
        checkpoint_state = {'epoch': epoch + 1,
                            'state_dict': self.model.state_dict(),
                            'optimizer': self.optimizer.state_dict(), }
        return checkpoint_state
