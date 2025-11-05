import os
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



class IterationEarlyStopping:
    def __init__(self, patience_epochs=10, patience_iterations=1000, min_iterations=500, 
                 verbose=True, delta=0, mode='min'):
        """
        混合早停机制：iteration级别判断 + epoch级别保存
        
        Args:
            patience_epochs: epoch级别的容忍数
            patience_iterations: iteration级别的容忍数  
            min_iterations: 最少训练iteration数（避免过早停止）
            verbose: 是否打印信息
            delta: 最小改善阈值
            mode: 'min'表示最小化损失，'max'表示最大化指标
        """
        self.patience_epochs = patience_epochs
        self.patience_iterations = patience_iterations
        self.min_iterations = min_iterations
        self.verbose = verbose
        self.delta = delta
        self.mode = mode
        
        self.counter_epochs = 0
        self.counter_iterations = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.global_iteration = 0
        
        # 用于iteration级别监控的滑动窗口
        self.loss_window = []
        self.window_size = 100  # 监控最近100个iteration的损失

    def __call__(self, current_loss, model,  model_path=None , model_name=None, is_iteration=False, current_iteration=0):
        """
        调用早停判断
        
        Args:
            current_loss: 当前损失值
            model: 模型
            path: 保存路径
            is_iteration: 是否为iteration级别调用
            current_iteration: 当前全局iteration数
        """
        self.global_iteration = current_iteration
        
        if self.mode == 'min':
            score = -current_loss
        else:
            score = current_loss
        
        # 第一次调用
        if self.best_score is None:
            self.best_score = score
            if model_path is not None and not is_iteration and model_name is not None:  # 只在epoch级别保存
                self.save_checkpoint(current_loss, model, model_path,model_name)
            return
        
        # 判断是否改善
        improved = score > self.best_score + self.delta
        
        if improved:
            # 有改善：更新最佳分数，重置计数器
            self.best_score = score
            self.counter_epochs = 0
            self.counter_iterations = 0
            if model_path is not None and not is_iteration and model_name is not None:  # 只在epoch级别保存模型
                self.save_checkpoint(current_loss, model, model_path,model_name)
        else:
            # 无改善：增加计数器
            if is_iteration:
                self.counter_iterations += 1
            else:
                self.counter_epochs += 1
        
        # 判断是否早停
        stop_by_epoch = self.counter_epochs >= self.patience_epochs
        stop_by_iteration = (self.counter_iterations >= self.patience_iterations and 
                           self.global_iteration >= self.min_iterations)
        
        if stop_by_epoch or stop_by_iteration:
            self.early_stop = True
            if self.verbose:
                if stop_by_epoch:
                    print(f'Early stopping: {self.counter_epochs} epochs without improvement')
                else:
                    print(f'Early stopping: {self.counter_iterations} iterations without improvement '
                          f'(total: {self.global_iteration} iterations)')

    def update_loss_window(self, loss):
        """更新损失滑动窗口"""
        self.loss_window.append(loss)
        if len(self.loss_window) > self.window_size:
            self.loss_window.pop(0)
    
    def is_loss_stable(self, threshold=0.001):
        """判断损失是否趋于稳定"""
        if len(self.loss_window) < self.window_size:
            return False
        
        recent_losses = np.array(self.loss_window)
        # 计算最近窗口内损失的变异系数
        if np.mean(recent_losses) > 0:
            cv = np.std(recent_losses) / np.mean(recent_losses)
            return cv < threshold
        return False

    def save_checkpoint(self, val_loss, model, model_path,model_name):
        """保存模型检查点"""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            torch.save(model.state_dict(), f"{model_path}/{model_name}.pth")
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


