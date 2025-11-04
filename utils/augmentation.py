import numpy as np
import random
import torch


def jitter(x, sigma=0.03):
    # https://arxiv.org/pdf/1706.00527.pdf
    # 抖动：向时间序列数据中添加高斯噪声，以模拟噪声数据。
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)

def scaling(x, sigma=0.1):
    # https://arxiv.org/pdf/1706.00527.pdf
    # 缩放：随机缩放数据，每个特征乘以一个随机因子
    factor = np.random.normal(loc=1., scale=sigma, size=(x.shape[0],x.shape[2]))
    return np.multiply(x, factor[:,np.newaxis,:])

def shift(x, sigma=0.1):
    """
    平移：将时间序列数据向前或向后平移一个随机的时间步长。
    :param x: 输入时间序列数据，形状为(T, C)
    :param p: 平移概率
    :param sigma: 平移的高斯噪声强度
    :return: 平移后的时间序列数据
    """
    return x + (torch.randn(x.shape[-1]) * sigma).numpy()

def wdba(x, labels, batch_size=6, slope_constraint="symmetric", use_window=True, verbose=0):
    """
    基于动态时间规整（DTW）的加权数据库平均（WDBA）算法。
    
    该函数旨在通过DTW计算在时间序列中找到相似的子序列，并对它们进行加权平均，
    用于数据平滑或聚类中心的计算。
    
    参数:
    - x: 二维数组，包含多个时间序列。
    - labels: 一维数组，对应于每个时间序列的标签，用于选择相同标签的序列进行处理。
    - batch_size: 整数，每次处理的子序列数量。
    - slope_constraint: 字符串，指定DTW计算中的斜率约束，可选值为"symmetric"或"asymmetric"。
    - use_window: 布尔值，决定是否使用DTW计算中的滑动窗口。
    - verbose: 整数，控制输出的详细程度，0为不输出，-1为关闭警告。
    
    返回:
    - 二维数组，包含经过WDBA处理后的时间序列。
    """
    # 将输入x转换为numpy数组
    # https://ieeexplore.ieee.org/document/8215569
    # use verbose = -1 to turn off warnings    
    # slope_constraint is for DTW. "symmetric" or "asymmetric"
    # 通过DTW平均生成新样本，可以较好地保持趋势和季节性。
    x = np.array(x)
    # 导入DTW模块
    import utils.dtw as dtw
    
    # 根据参数决定是否使用滑动窗口
    if use_window:
        window = np.ceil(x.shape[1] / 10.).astype(int)
    else:
        window = None
    
    # 获取原始序列的索引
    orig_steps = np.arange(x.shape[1])
    # 获取标签数组中的最大值索引，用于后续选择相同标签的序列
    l = np.argmax(labels, axis=1) if labels.ndim > 1 else labels
        
    # 初始化输出数组，与输入x相同形状
    ret = np.zeros_like(x)
    # 遍历每个时间序列
    # for i in tqdm(range(ret.shape[0])):
    for i in range(ret.shape[0]):
        # 选择与当前序列标签相同的其他序列
        # get the same class as i
        choices = np.where(l == l[i])[0]
        # 如果有相同标签的序列
        if choices.size > 0:        
            # 从相同标签的序列中随机选择batch_size个作为原型
            # pick random intra-class pattern
            k = min(choices.size, batch_size)
            random_prototypes = x[np.random.choice(choices, k, replace=False)]
            
            # 初始化DTW矩阵，用于存储随机原型之间的DTW距离
            # calculate dtw between all
            dtw_matrix = np.zeros((k, k))
            # 计算所有随机原型之间的DTW距离
            for p, prototype in enumerate(random_prototypes):
                for s, sample in enumerate(random_prototypes):
                    if p == s:
                        dtw_matrix[p, s] = 0.
                    else:
                        dtw_matrix[p, s] = dtw.dtw(prototype, sample, dtw.RETURN_VALUE, slope_constraint=slope_constraint, window=window)
                        
            # 找到DTW距离总和最小的原型作为中位数
            # get medoid
            medoid_id = np.argsort(np.sum(dtw_matrix, axis=1))[0]
            nearest_order = np.argsort(dtw_matrix[medoid_id])
            medoid_pattern = random_prototypes[medoid_id]
            
            # 计算加权平均
            # start weighted DBA
            average_pattern = np.zeros_like(medoid_pattern)
            weighted_sums = np.zeros((medoid_pattern.shape[0]))
            # 遍历所有原型，计算加权和
            for nid in nearest_order:
                if nid == medoid_id or dtw_matrix[medoid_id, nearest_order[1]] == 0.:
                    average_pattern += medoid_pattern
                    weighted_sums += np.ones_like(weighted_sums) 
                else:
                    # 计算中位数与当前原型之间的DTW路径
                    path = dtw.dtw(medoid_pattern, random_prototypes[nid], dtw.RETURN_PATH, slope_constraint=slope_constraint, window=window)
                    dtw_value = dtw_matrix[medoid_id, nid]
                    warped = random_prototypes[nid, path[1]]
                    # 计算权重
                    weight = np.exp(np.log(0.5)*dtw_value/dtw_matrix[medoid_id, nearest_order[1]])
                    # 根据权重更新加权和
                    average_pattern[path[0]] += weight * warped
                    weighted_sums[path[0]] += weight 
            
            # 计算加权平均值
            ret[i,:] = average_pattern / weighted_sums[:,np.newaxis]
        else:
            # 如果没有相同标签的序列，直接输出原序列
            ret[i,:] = x[i]
    return ret

# ========================================================
# 逆序
def reverse_order(x):
    return torch.flip(x, dims=[1])

def detrend(x):
    from scipy.signal import detrend as scipy_detrend
    x_numpy = x.cpu().numpy()  # 将张量转换为NumPy数组
    detrended = scipy_detrend(x_numpy, axis=1) 
    # 将 NumPy 数组转换回 PyTorch 张量
    return torch.tensor(detrended, dtype=x.dtype, device=x.device)
    
# 累积和
def cumulative_sum(x):
    return torch.cumsum(x, dim=1)  # 使用torch.cumsum

# 多项式变换
def polynomial_transform(x, degree=2):
    return torch.pow(x, degree)  # 使用torch.pow

def augment(x, y,negative_num,file_path,plot=True):
    import matplotlib.pyplot as plt
    import os
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
        else :
            x_augment_n = polynomial_transform(x)
        
            
        x_augment_n_list.append(x_augment_n)
      
    # 可视化原始数据和增强数据
    batch_size, seq_len, num_features = x.shape

    if plot:
        # 确保plot文件夹存在
        plot_dir = os.path.join(file_path, 'augment_plot')
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
            for i in range(num_features):
                for j in range(negative_num):
                    plt.figure(figsize=(15, 5))
                    plt.plot(x[0, :, i], label='Original', linestyle='-', marker='o')
                    plt.plot(x_augment_p[0, :, i], label='Positive Augment', linestyle='--', marker='x')
                    plt.plot(x_augment_n_list[j][0, :, i], label='Negative Augment', linestyle=':', marker='s')
                    plt.title(f'Feature {i + 1} Negative Augment {j + 1}')
                    plt.xlabel('Time')
                    plt.ylabel('Value')
                    plt.legend()
                    plt.show() 
                    plt.savefig(os.path.join(plot_dir, f'feature_{i + 1}_negative_{j +1}_comparison.png'))
                    plt.close()
    return x,x_augment_p,x_augment_n_list