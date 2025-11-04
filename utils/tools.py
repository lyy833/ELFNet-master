import torch.nn.functional as F

import numpy as np
import torch
import matplotlib.pyplot as plt
import math
from utils.augmentation import *

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, epoch, args):
    """
    根据不同的策略调整学习率。
    
    本函数根据当前的训练周期（epoch）和配置参数（args）来动态调整优化器（optimizer）的学习率。
    支持三种类型的学习率调整策略：'type1', 'type2', 和 'cosine'，通过args.lradj参数指定。
    
    参数:
    optimizer: 优化器，其学习率需要调整。
    epoch: 当前的训练周期数。
    args: 包含调整策略和可能的其他相关配置的参数。
    
    返回:
    无返回值，直接修改优化器的学习率并打印更新后的学习率。
    """
    # 根据'args.lradj'参数值选择不同的学习率调整策略
    if args.lradj == 'type1':
        # 按照每1个周期学习率减半的策略调整学习率
        lr_adjust = {epoch: args.lr * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        # 按照预设的周期点降低学习率
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == "cosine":
        # 按照余弦退火策略调整学习率
        lr_adjust = {epoch: args.lr / 2 * (1 + math.cos(epoch / args.train_epochs * math.pi))}
    
    # 检查当前周期是否在调整字典中，如果是，则更新学习率
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        # 更新优化器的学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        # 打印学习率更新信息
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience, verbose=True, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model, path=None):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            if path is not None:
                self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            #if self.verbose:
            #    print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience :
                self.early_stop = True
                print('Early stopping')
        else:
            self.best_score = score
            if path is not None:
                self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            torch.save(model.state_dict(), path  +'/'+ 'model.pth')
        self.val_loss_min = val_loss



class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


