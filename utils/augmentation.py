import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
import os


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

def reverse_order(x):
    """
    逆序
    """
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

def cumulative_sum(x):
    """
    累积和
    """
    if isinstance(x, torch.Tensor):
        return torch.cumsum(x, dim=1)
    else:
        return np.cumsum(x, axis=1)

def polynomial_transform(x, degree=2):
    """
    多项式变换
    """
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



class VariableImportanceAnalyzer:
    """关键变量自动识别器，用于因果感知增强器(CausalAwareAugmenter)"""
    
    def __init__(self, target_index=0, max_lag=24):
        self.target_index = target_index
        self.max_lag = max_lag
    
    def analyze_variable_importance(self, data_x, data_y):
        """
        基于多种指标分析变量重要性
        data_x: [seq_len, n_vars]
        data_y: [seq_len] 目标变量（负荷）
        """
        n_vars = data_x.shape[1]
        importance_scores = np.zeros(n_vars)
        
        for var_idx in range(n_vars):
            if var_idx == self.target_index:
                continue
                
            var_data = data_x[:, var_idx]
            
            try:
                # 1. 互信息
                mi_score = self._mutual_info_score(var_data, data_y)
                
                # 2. 皮尔逊相关系数
                corr_score = abs(self._pearson_corr(var_data, data_y))
                
                # 3. 格兰杰因果关系（简化版）
                granger_score = self._simplified_granger_test(var_data, data_y)
                
                # 4. 时序滞后相关性
                lag_corr_score = self._lag_correlation(var_data, data_y)
                
                # 综合评分
                combined_score = (0.4 * mi_score + 0.3 * corr_score + 
                                0.2 * granger_score + 0.1 * lag_corr_score)
                importance_scores[var_idx] = combined_score
                
            except Exception as e:
                print(f"Error analyzing variable {var_idx}: {e}")
                importance_scores[var_idx] = 0.0
        
        return importance_scores
    
    def _mutual_info_score(self, x, y):
        """计算互信息分数"""
        # 确保数据是1D的
        x_1d = x.reshape(-1, 1) if len(x.shape) == 1 else x
        y_1d = y.flatten()
        
        # 使用sklearn的互信息回归
        mi = mutual_info_regression(x_1d, y_1d, random_state=42)[0]
        
        # 归一化到0-1范围
        mi_normalized = min(mi / (np.std(x) * np.std(y) + 1e-8), 1.0)
        return mi_normalized
    
    def _pearson_corr(self, x, y):
        """计算皮尔逊相关系数"""
        # 处理可能的常数序列
        if np.std(x) == 0 or np.std(y) == 0:
            return 0.0
            
        corr_coef = np.corrcoef(x, y)[0, 1]
        return corr_coef if not np.isnan(corr_coef) else 0.0
    
    def _simplified_granger_test(self, x, y, max_lag=5):
        """
        简化的格兰杰因果关系检验
        使用线性回归比较包含/不包含x滞后的预测效果
        """
        n = len(x)
        if n < max_lag * 3:  # 样本太少
            return 0.0
        
        try:
            from sklearn.linear_model import LinearRegression
            
            # 准备特征矩阵
            X_lagged = np.zeros((n - max_lag, max_lag * 2))
            y_target = y[max_lag:]
            
            for i in range(max_lag):
                X_lagged[:, i] = x[max_lag - i - 1:n - i - 1]  # x的滞后
                X_lagged[:, max_lag + i] = y[max_lag - i - 1:n - i - 1]  # y的滞后
            
            # 模型1: 只包含y的滞后
            X_model1 = X_lagged[:, max_lag:]
            model1 = LinearRegression()
            model1.fit(X_model1, y_target)
            r2_model1 = model1.score(X_model1, y_target)
            
            # 模型2: 包含x和y的滞后
            X_model2 = X_lagged
            model2 = LinearRegression()
            model2.fit(X_model2, y_target)
            r2_model2 = model2.score(X_model2, y_target)
            
            # 计算改进程度
            improvement = max(0, r2_model2 - r2_model1)
            return improvement
            
        except Exception as e:
            print(f"Granger test failed: {e}")
            return 0.0
    
    def _lag_correlation(self, x, y, max_lag=None):
        """计算最大时序滞后相关性"""
        if max_lag is None:
            max_lag = min(self.max_lag, len(x) // 4)
        
        # 确保序列长度足够
        if len(x) < max_lag * 2:
            return 0.0
        
        max_corr = 0.0
        
        # 检查正负滞后
        for lag in range(-max_lag, max_lag + 1):
            if lag == 0:
                # 零滞后相关性
                x_shifted = x
                y_shifted = y
            elif lag > 0:
                # 正滞后：x领先y
                x_shifted = x[:-lag] if lag < len(x) else x
                y_shifted = y[lag:] if lag < len(y) else y
            else:
                # 负滞后：y领先x
                lag_abs = abs(lag)
                x_shifted = x[lag_abs:] if lag_abs < len(x) else x
                y_shifted = y[:-lag_abs] if lag_abs < len(y) else y
            
            # 确保长度匹配
            min_len = min(len(x_shifted), len(y_shifted))
            if min_len < 10:  # 样本太少
                continue
                
            x_shifted = x_shifted[:min_len]
            y_shifted = y_shifted[:min_len]
            
            # 计算相关性
            if np.std(x_shifted) == 0 or np.std(y_shifted) == 0:
                continue
                
            corr = abs(np.corrcoef(x_shifted, y_shifted)[0, 1])
            if not np.isnan(corr):
                max_corr = max(max_corr, corr)
        
        return max_corr
    
    def select_causal_variables(self, importance_scores, top_k=None, threshold=None):
        """根据评分选择关键变量"""
        if top_k is None:
            # 自动确定top_k：选择评分明显高于平均的变量
            mean_score = np.mean(importance_scores)
            std_score = np.std(importance_scores)
            threshold_auto = mean_score + 0.5 * std_score
            
            causal_vars = [i for i, score in enumerate(importance_scores) 
                          if score > threshold_auto and i != self.target_index]
            
            # 确保至少选择1个，最多选择一半的变量
            if len(causal_vars) == 0:
                # 如果没有明显重要的，选择评分最高的1-2个
                causal_vars = [np.argmax(importance_scores)]
            elif len(causal_vars) > len(importance_scores) // 2:
                # 如果太多，选择评分最高的前1/3
                sorted_indices = np.argsort(importance_scores)[::-1]
                causal_vars = sorted_indices[:max(1, len(importance_scores) // 3)].tolist()
            
            non_causal_vars = [i for i in range(len(importance_scores)) 
                              if i not in causal_vars and i != self.target_index]
        
        else:
            # 使用指定的top_k
            sorted_indices = np.argsort(importance_scores)[::-1]
            causal_vars = [idx for idx in sorted_indices[:top_k] if idx != self.target_index]
            non_causal_vars = [idx for idx in sorted_indices[top_k:] if idx != self.target_index]
        
        print(f"变量重要性评分: {importance_scores}")
        print(f"自动识别关键变量: {causal_vars}")
        print(f"非关键变量: {non_causal_vars}")
        
        return causal_vars, non_causal_vars


class CausalAwareAugmenter:
    """因果感知增强器"""
    
    def __init__(self, causal_vars, non_causal_vars):
        self.causal_vars = causal_vars
        self.non_causal_vars = non_causal_vars
    
    def augment_batch(self, batch_x):
        """
        主入口函数 - 对批次数据进行因果感知增强
        返回: (原始样本, 正样本, 负样本列表)
        """
        # 生成正样本 - 保持因果结构
        positive_x = self.causal_positive_augment(batch_x)
        
        # 生成负样本列表 - 破坏因果结构
        negative_x_list = self.causal_negative_augment(batch_x)
        
        return batch_x, positive_x, negative_x_list,self.causal_vars,self.non_causal_vars


    def causal_positive_augment(self, x):
        """
        因果保持的正增强 - 核心思想：只扰动非因果变量，保持因果变量不变
        """
        x_aug = x.clone()
        
        # 只对非因果变量进行多样化扰动
        for var_idx in self.non_causal_vars:
            # 随机选择扰动类型
            aug_type = np.random.choice(['scale', 'shift', 'noise', 'combination'])
            
            if aug_type == 'scale':
                scale = 0.9 + torch.rand(1).to(x.device) * 0.2
                x_aug[:, :, var_idx] = x[:, :, var_idx] * scale
            elif aug_type == 'shift':
                shift = torch.randn(1).to(x.device) * 0.05
                x_aug[:, :, var_idx] = x[:, :, var_idx] + shift
            elif aug_type == 'noise':
                noise = torch.randn_like(x[:, :, var_idx]) * 0.02
                x_aug[:, :, var_idx] = x[:, :, var_idx] + noise
            else:  # combination
                scale = 0.95 + torch.rand(1).to(x.device) * 0.1
                noise = torch.randn_like(x[:, :, var_idx]) * 0.01
                x_aug[:, :, var_idx] = x[:, :, var_idx] * scale + noise
        
        # 关键：因果变量完全保持不变
        return x_aug
    
    def causal_negative_augment(self, x):
        """
        因果破坏的负增强 - 核心思想：专门针对因果变量(即自动识别的关键变量)进行破坏性干预
        """
        neg_samples = []
        
        # 策略1: 因果变量置换（最有效的策略）
        neg_samples.append(self._causal_variable_permutation(x))
        
        # 策略2: 因果变量独立扰动
        neg_samples.append(self._causal_variable_corruption(x))
        
        # 策略3: 因果关系破坏（交换因果变量的时序模式）
        neg_samples.append(self._causal_relationship_disruption(x))
        
        return neg_samples
    
    def _causal_variable_permutation(self, x):
        """因果变量置换 - 在不同样本间交换同一种因果变量"""
        x_neg = x.clone()
        batch_size = x.size(0)
        
        if batch_size > 1:  # 需要至少2个样本来置换
            perm_indices = torch.randperm(batch_size)
            for var_idx in self.causal_vars:
                x_neg[:, :, var_idx] = x[perm_indices, :, var_idx]
        
        return x_neg
    
    def _causal_variable_corruption(self, x):
        """因果变量独立扰动 - 对每个因果变量独立施加强扰动"""
        x_neg = x.clone()
        B, T, C = x.shape
        
        for var_idx in self.causal_vars:
            # 对每个因果变量独立施加扰动
            aug_type = np.random.choice(['strong_scale', 'strong_shift', 'pattern_noise'])
            
            if aug_type == 'strong_scale':
                # 强缩放：0.5-1.5倍
                scale = 0.5 + torch.rand(B, 1).to(x.device)  # 每个样本不同的缩放
                x_neg[:, :, var_idx] = x[:, :, var_idx] * scale
            elif aug_type == 'strong_shift':
                # 强偏移：±0.3倍标准差
                var_std = x[:, :, var_idx].std()
                shift = torch.randn(B, 1).to(x.device) * 0.3 * var_std
                x_neg[:, :, var_idx] = x[:, :, var_idx] + shift
            else:  # pattern_noise
                # 模式噪声：破坏时序模式
                noise_std = 0.2 * x[:, :, var_idx].std()
                pattern_noise = torch.randn(B, T).to(x.device) * noise_std
                x_neg[:, :, var_idx] = x[:, :, var_idx] + pattern_noise
        
        return x_neg
    
    def _causal_relationship_disruption(self, x):
        """因果关系破坏 - 样本内交换因果变量"""
        x_neg = x.clone()
        B, T, C = x.shape
        
        if len(self.causal_vars) >= 2:
            # 随机选择两个因果变量交换它们的时序模式
            var1, var2 = np.random.choice(self.causal_vars, 2, replace=False)
            
            # 交换时序模式（保持各自的幅度特性）
            var1_data = x[:, :, var1].clone()
            var2_data = x[:, :, var2].clone()
            
            # 标准化后交换
            var1_mean, var1_std = var1_data.mean(), var1_data.std()
            var2_mean, var2_std = var2_data.mean(), var2_data.std()
            
            if var1_std > 1e-6 and var2_std > 1e-6:
                # 标准化交换后再还原
                var1_normalized = (var1_data - var1_mean) / var1_std
                var2_normalized = (var2_data - var2_mean) / var2_std
                
                x_neg[:, :, var1] = var2_normalized * var1_std + var1_mean
                x_neg[:, :, var2] = var1_normalized * var2_std + var2_mean
        
        return x_neg


class DynamicPeakDetectionAugmenter:
    """
    动态峰谷检测增强器,专门针对负荷变量进行峰谷检测,
    使用分位数阈值识别高负荷和低负荷时段,基于电力系统实际的峰谷用电模式进行增强
    """
    def __init__(self, peak_percentile=0.85, off_peak_percentile=0.15, 
                 min_peak_ratio=1.2, max_augment_strength=0.15):
        self.peak_percentile = peak_percentile # 波峰分位数
        self.off_peak_percentile = off_peak_percentile # 波谷分位数
        self.min_peak_ratio = min_peak_ratio  # 峰谷最小比例阈值
        self.max_augment_strength = max_augment_strength # 最大增强强度
    
    def dynamic_peak_augment(self, x, load_var_index):
        """
        动态峰谷检测和增强。
        - 波峰衰减：模拟需求响应、能效措施、分布式发电
        - 波谷加噪：模拟基础负荷波动、小用户随机行为、测量噪声
        """
        x_aug = x.clone()
        B, T, C = x.shape
        
        load_data = x[:, :, load_var_index]  # [B, T]
        
        for i in range(B): # 依次处理每个样本
            sample_load = load_data[i] # [1,T]
            
            # 计算峰谷阈值
            peak_threshold = torch.quantile(sample_load, self.peak_percentile) # 默认sample_load升序排列后T*85%位置对应元素的值
            off_peak_threshold = torch.quantile(sample_load, self.off_peak_percentile) # # 默认sample_load升序排列后T*15%位置对应元素的值
            
            # 峰谷差异显著性检查，避免对无明显峰谷模式的序列进行无效增强
            peak_valley_ratio = peak_threshold / (off_peak_threshold + 1e-6) 
            if peak_valley_ratio < self.min_peak_ratio: # 峰谷差异不明显，跳过增强
                continue
            
            peak_mask = sample_load > peak_threshold
            off_peak_mask = sample_load < off_peak_threshold
            
            # 峰时段增强
            if peak_mask.any():
                # max_augment_strength用于最大增强强度限制，防止过度增强导致数据失真
                peak_scale = 1.0 - torch.rand(1).to(x.device) * self.max_augment_strength
                x_aug[i, peak_mask, :] = x_aug[i, peak_mask, :] * peak_scale
            
            # 谷时段增强
            if off_peak_mask.any():
                noise_std = torch.rand(1).to(x.device) * self.max_augment_strength * 0.1
                noise = torch.randn_like(x_aug[i, off_peak_mask, :]) * noise_std
                x_aug[i, off_peak_mask, :] = x_aug[i, off_peak_mask, :] + noise
        
        return x_aug


class TemporalAugmenter:
    """
    通用时序局部模式增强器。
    - 对所有变量进行基于序列形态的增强
    - 通过局部极值检测和波动性分析识别时序模式
    - 捕捉序列的通用时序特征，不限于特定领域
    """
    def __init__(self):
        pass
    
    def temporal_pattern_augment(self, x):
        x_aug = x.clone()
        B, T, C = x.shape
        
        # 方法1：基于局部极值的增强
        x_aug = self._local_extremum_augment(x_aug)
        
        # 方法2：基于波动模式的增强
        x_aug = self._fluctuation_pattern_augment(x_aug)
        
        return x_aug
    
    def _local_extremum_augment(self, x):
        """
        基于局部极值点的增强
        """
        B, T, C = x.shape
        
        for i in range(B): # 依次处理每个样本
            for var_idx in range(C): # 每个样本中依次处理每个变量
                var_series = x[i, :, var_idx]
                
                # 检测局部极值
                diff = torch.diff(var_series, prepend=var_series[0:1])
                diff_sign = torch.sign(diff)
                diff_sign_change = torch.diff(diff_sign, prepend=diff_sign[0:1])
                
                # 极值点：符号变化非零的位置
                extremum_mask = diff_sign_change != 0
                
                if extremum_mask.any():
                    # 对极值点进行轻微扰动
                    extremum_scale = 0.98 + torch.rand(1).to(x.device) * 0.04 # 缩放因子范围为 [0.98，1.02]
                    x[i, extremum_mask, var_idx] = x[i, extremum_mask, var_idx] * extremum_scale
        
        return x
    
    def _fluctuation_pattern_augment(self, x):
        """
        基于波动模式的增强
        """
        B, T, C = x.shape
        
        for i in range(B):
            # 计算序列的波动性
            # x[i]形状为[T,C],因此 volatility形状为[C]
            # 通过样本x[i]每个变量的标准差反映每个变量波动性
            volatility = torch.std(x[i], dim=0)  
            
            for var_idx in range(C):
                if volatility[var_idx] > 0.01:  # 只对波动性较大的变量增强
                    # 根据波动性调整增强强度
                    aug_strength = min(volatility[var_idx].item() * 5, 0.1)
                    noise = torch.randn(T).to(x.device) * aug_strength
                    x[i, :, var_idx] = x[i, :, var_idx] + noise
        
        return x
    
class PatternBasedHolidayAugmenter:
    """
    基于序列模式识别的节假日增强器
    """
    def __init__(self, weekend_likelihood_threshold=0.7):
        self.weekend_likelihood_threshold = weekend_likelihood_threshold
    
    def pattern_based_holiday_augment(self, x, load_var_index=0):
        """
        基于负荷模式识别"疑似节假日"并进行增强
        """
        x_aug = x.clone()
        B, T, C = x.shape
        
        for i in range(B):
            # 检测是否具有节假日/周末模式
            is_holiday_like = self._detect_holiday_pattern(x[i, :, load_var_index])
            
            if is_holiday_like:
                # 应用节假日增强
                holiday_scale = 0.7 + torch.rand(1).to(x.device) * 0.6
                x_aug[i, :, :] = x_aug[i, :, :] * holiday_scale
        
        return x_aug
    
    def _detect_holiday_pattern(self, load_series):
        """
        检测负荷序列是否具有节假日模式
        """
        T = load_series.size(0)
        
        # 特征1：日间波动性较低
        daily_volatility = torch.std(load_series)
        
        # 特征2：负荷水平相对平稳
        max_load = torch.max(load_series)
        min_load = torch.min(load_series)
        load_range_ratio = (max_load - min_load) / max_load
        
        # 特征3：早晚高峰不明显（简化检测）
        if T >= 24:  # 假设序列包含完整的一天
            morning_hours = load_series[8:10] if T >= 10 else load_series[-10:-8]
            evening_hours = load_series[18:20] if T >= 20 else load_series[-20:-18]
            day_avg = load_series.mean()
            
            morning_peak = morning_hours.max() / day_avg
            evening_peak = evening_hours.max() / day_avg
            
            # 综合判断
            is_holiday_like = (daily_volatility < 0.1 and 
                             load_range_ratio < 0.5 and 
                             morning_peak < 1.3 and 
                             evening_peak < 1.3)
        else:
            # 对于较短序列，使用简化判断
            is_holiday_like = (daily_volatility < 0.08 and 
                             load_range_ratio < 0.4)
        
        return is_holiday_like

class DomainAugmentationFramework:
    """
    基于领域知识的数据增强框架
    """
    def __init__(self,  target_index):
        self.target_index = target_index
        self.importance_analyzer = VariableImportanceAnalyzer(target_index)
        self.causal_augmenter = None
        self.peak_augmenter = DynamicPeakDetectionAugmenter()
        self.temporal_augmenter = TemporalAugmenter()
        self.holiday_augmenter = PatternBasedHolidayAugmenter()

        
    def initialize_from_data(self, data_x, data_y):
        """初始化因果感知增强器"""
        importance_scores = self.importance_analyzer.analyze_variable_importance(data_x, data_y)
        causal_vars, non_causal_vars = self.importance_analyzer.select_causal_variables(importance_scores)
        self.causal_augmenter = CausalAwareAugmenter(causal_vars, non_causal_vars)
        return causal_vars, non_causal_vars
    
    def augment_batch(self, batch_x, batch_y,plot_dir=None, plot_augment_flag=False):
        """批处理增强 - 各司其职"""
        # 1. 正负增强：因果感知增强（核心）
        batch_x, positive_x, negative_x_list,causal_vars,non_causal_vars = self.causal_augmenter.augment_batch(batch_x)
        self.causal_vars = causal_vars
        self.non_causal_vars = non_causal_vars
        
        # 2. 正增强：峰谷增强（负荷变量）
        positive_x = self.peak_augmenter.dynamic_peak_augment(positive_x, self.target_index)
        
        # 3. 正增强：节假日增强
        positive_x = self.holiday_augmenter.pattern_based_holiday_augment(positive_x, self.target_index)
        
        # 4. 正负增强：通用时序局部模式增强
        positive_x = self.temporal_augmenter.temporal_pattern_augment(positive_x)
        
        # 对负样本也应用通用时序局部模式增强以增加多样性
        enhanced_negative_list = []
        for neg_x in negative_x_list:
            neg_enhanced = self.temporal_augmenter.temporal_pattern_augment(neg_x)
            enhanced_negative_list.append(neg_enhanced)
        
        # 可视化
        if plot_augment_flag and plot_dir:
            self.visualize_augmentations(
                batch_x, positive_x, enhanced_negative_list,
                self.causal_vars, self.non_causal_vars,
                plot_dir, plot_augment_flag, True
            )

        return batch_x, positive_x, enhanced_negative_list
        
    def visualize_augmentations(self, original, positive, negatives, causal_vars, non_causal_vars, 
                           plot_dir, plot_augment_flag=True, plot_augment=True):
        """
        可视化原始数据和增强数据
        original: [B, T, C] 原始数据
        positive: [B, T, C] 正样本
        negatives: list of [B, T, C] 负样本列表
        causal_vars: 因果变量索引列表
        non_causal_vars: 非因果变量索引列表
        """
        if not plot_augment_flag or not plot_augment:
            return
        
        batch_size, seq_len, num_features = original.shape
        negative_num = len(negatives)
        
        # 只可视化第一个样本
        sample_idx = 0
        
        for i in range(num_features):
            # 确定变量类型用于标签
            var_type = "因果" if i in causal_vars else "非因果"
            
            for j in range(negative_num):
                plt.figure(figsize=(15, 5))
                
                # 提取数据 - 注意维度是 [B, T, C]
                orig_data = original[sample_idx, :, i].detach().cpu().numpy()
                pos_data = positive[sample_idx, :, i].detach().cpu().numpy()
                neg_data = negatives[j][sample_idx, :, i].detach().cpu().numpy()
                
                # 绘制三条线
                plt.plot(orig_data, label='原始数据', linestyle='-', marker='o', markersize=3)
                plt.plot(pos_data, label='正增强', linestyle='--', marker='x', markersize=3)
                plt.plot(neg_data, label=f'负增强{j+1}', linestyle=':', marker='s', markersize=3)
                
                plt.title(f'{var_type}变量 {i} - 负增强策略 {j+1}')
                plt.xlabel('时间步')
                plt.ylabel('数值')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # 保存图片
                os.makedirs(plot_dir, exist_ok=True)
                plt.savefig(os.path.join(plot_dir, f'var_{i}_causal_{i in causal_vars}_neg_{j+1}.png'), 
                        dpi=150, bbox_inches='tight')
                plt.close()

