import numpy as np
import torch


def jitter(x, sigma=0.03):
    # https://arxiv.org/pdf/1706.00527.pdf
    # 抖动：向时间序列数据中添加高斯噪声，以模拟噪声数据。
    if isinstance(x, torch.Tensor): # 如果输入是 torch.Tensor，则在相同设备和 dtype 上使用 torch.randn_like；
        noise = torch.randn_like(x) * sigma
        return x + noise
    else:  # 否则使用 numpy 生成噪声。
        return x + np.random.normal(loc=0., scale=sigma, size=x.shape)

def scaling(x, sigma=0.1):
    # https://arxiv.org/pdf/1706.00527.pdf
    # 缩放：随机缩放数据，每个特征乘以一个随机因子
    if isinstance(x, torch.Tensor):
        # factor shape: (batch, features)
        factor = torch.normal(mean=1.0, std=sigma, size=(x.shape[0], x.shape[2]), device=x.device, dtype=x.dtype)
        factor = factor.unsqueeze(1)  # (batch, 1, features)
        return x * factor
    else:
        factor = np.random.normal(loc=1., scale=sigma, size=(x.shape[0], x.shape[2]))
        return np.multiply(x, factor[:, np.newaxis, :])


def shift(x, sigma=0.1):
    """
    平移/微扰：为时间序列添加小的随机偏移（高斯噪声）
    兼容 torch.Tensor 与 numpy.ndarray
    """
    if isinstance(x, torch.Tensor):
        noise = torch.randn(size=x.shape, device=x.device, dtype=x.dtype) * sigma
        return x + noise
    else:
        return x + np.random.normal(loc=0., scale=sigma, size=x.shape)


def wdba(x, labels, batch_size=6, slope_constraint="symmetric", use_window=True, verbose=0):
    """
    基于动态时间规整（DTW）的加权数据库平均（WDBA）算法。
    兼容 torch.Tensor 与 numpy.ndarray，返回与输入相同的类型与设备。
    """
    # detect original type/device/dtype
    input_is_tensor = isinstance(x, torch.Tensor)
    x_device = None
    x_dtype = None
    if input_is_tensor:
        x_device = x.device
        x_dtype = x.dtype
        x_np = x.detach().cpu().numpy()
    else:
        x_np = np.array(x)

    if isinstance(labels, torch.Tensor):
        labels_np = labels.detach().cpu().numpy()
    else:
        labels_np = np.array(labels)

    # https://ieeexplore.ieee.org/document/8215569
    import utils.dtw as dtw

    if use_window:
        window = np.ceil(x_np.shape[1] / 10.).astype(int)
    else:
        window = None

    orig_steps = np.arange(x_np.shape[1])
    l = np.argmax(labels_np, axis=1) if labels_np.ndim > 1 else labels_np

    ret = np.zeros_like(x_np)
    for i in range(ret.shape[0]):
        choices = np.where(l == l[i])[0]
        if choices.size > 0:
            k = min(choices.size, batch_size)
            random_prototypes = x_np[np.random.choice(choices, k, replace=False)]

            dtw_matrix = np.zeros((k, k))
            for p, prototype in enumerate(random_prototypes):
                for s, sample in enumerate(random_prototypes):
                    if p == s:
                        dtw_matrix[p, s] = 0.
                    else:
                        dtw_matrix[p, s] = dtw.dtw(prototype, sample, dtw.RETURN_VALUE,
                                                   slope_constraint=slope_constraint, window=window)

            medoid_id = np.argsort(np.sum(dtw_matrix, axis=1))[0]
            nearest_order = np.argsort(dtw_matrix[medoid_id])
            medoid_pattern = random_prototypes[medoid_id]

            average_pattern = np.zeros_like(medoid_pattern)
            weighted_sums = np.zeros((medoid_pattern.shape[0]))
            for nid in nearest_order:
                if nid == medoid_id or dtw_matrix[medoid_id, nearest_order[1]] == 0.:
                    average_pattern += medoid_pattern
                    weighted_sums += np.ones_like(weighted_sums)
                else:
                    path = dtw.dtw(medoid_pattern, random_prototypes[nid], dtw.RETURN_PATH,
                                   slope_constraint=slope_constraint, window=window)
                    dtw_value = dtw_matrix[medoid_id, nid]
                    warped = random_prototypes[nid, path[1]]
                    weight = np.exp(np.log(0.5) * dtw_value / dtw_matrix[medoid_id, nearest_order[1]])
                    average_pattern[path[0]] += weight * warped
                    weighted_sums[path[0]] += weight

            # Avoid division by zero
            zero_mask = weighted_sums == 0
            weighted_sums[zero_mask] = 1.0
            ret[i, :] = average_pattern / weighted_sums[:, np.newaxis]
        else:
            ret[i, :] = x_np[i]

    if input_is_tensor:
        return torch.tensor(ret, dtype=x_dtype, device=x_device)
    else:
        return ret


# ========================================================
# 逆序
def reverse_order(x):
    if isinstance(x, torch.Tensor):
        return torch.flip(x, dims=[1])
    else:
        return np.flip(x, axis=1)


def detrend(x):
    from scipy.signal import detrend as scipy_detrend
    if isinstance(x, torch.Tensor):
        x_numpy = x.detach().cpu().numpy()
        detrended = scipy_detrend(x_numpy, axis=1)
        return torch.tensor(detrended, dtype=x.dtype, device=x.device)
    else:
        return scipy_detrend(np.array(x), axis=1)


# 累积和
def cumulative_sum(x):
    if isinstance(x, torch.Tensor):
        return torch.cumsum(x, dim=1)
    else:
        return np.cumsum(x, axis=1)


# 多项式变换
def polynomial_transform(x, degree=2):
    if isinstance(x, torch.Tensor):
        return torch.pow(x, degree)
    else:
        return np.power(x, degree)


def augment(x, y, negative_num, plot_dir,plot_augment, plot_augment_flag):
    import matplotlib.pyplot as plt
    import os

    def to_numpy(a):
        return a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else np.array(a)

    # 正增强
    x_jitter = jitter(x)
    x_scaling = scaling(x_jitter)
    x_wdba = wdba(x_scaling, y)
    x_shift = shift(x_wdba)
    x_augment_p = x_shift

    # 负增强
    x_augment_n_list = []
    for i in range(negative_num):
        if i % 4 == 0:
            x_augment_n = reverse_order(x)
        elif i % 4 == 1:
            x_augment_n = detrend(x)
        elif i % 4 == 2:
            x_augment_n = cumulative_sum(x)
        else:
            x_augment_n = polynomial_transform(x)

        x_augment_n_list.append(x_augment_n)

    # 可视化原始数据和增强数据
    batch_size, num_features, seq_len = x.shape

    if plot_augment_flag and plot_augment:
        for i in range(num_features):
            for j in range(negative_num):
                plt.figure(figsize=(15, 5))
                plt.plot(to_numpy(x)[0, i,:], label='Original', linestyle='-', marker='o')
                plt.plot(to_numpy(x_augment_p)[0, i,:], label='Positive Augment', linestyle='--', marker='x')
                plt.plot(to_numpy(x_augment_n_list[j])[0, i,:], label='Negative Augment', linestyle=':', marker='s')
                plt.title(f'Feature {i + 1} Negative Augment {j + 1}')
                plt.xlabel('Time')
                plt.ylabel('Value')
                plt.legend()
                plt.savefig(os.path.join(plot_dir, f'feature_{i + 1}_negative_{j +1}_comparison.png'))
                plt.close()
    return x, x_augment_p, x_augment_n_list