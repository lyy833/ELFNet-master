import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
import os

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
            causal_vars = [idx for idx in sorted_indices[:top_k] if idx != self.target_index] + [self.target_index]
            non_causal_vars = [idx for idx in sorted_indices[top_k:] if idx != self.target_index]
        
        print(f"变量重要性评分: {importance_scores}")
        print(f"自动识别关键变量: {causal_vars}")
        print(f"非关键变量: {non_causal_vars}")
        
        return causal_vars, non_causal_vars


class TrendSeasonalAugmenter:
    """针对趋势性或季节性单一成分的负增强器"""
    
    def __init__(self,freq):
        self.freq = freq
        # 趋势性破坏策略
        self.trend_corruption_methods = [
            self._add_linear_trend,
            self._reverse_trend,
            self._detrend_and_reshuffle
        ]
        
        # 季节性破坏策略  
        self.seasonal_corruption_methods = [
            self._phase_shift,
            self._seasonal_scale,
            self._periodic_noise
        ]
    
    def corrupt_for_trend(self, x):
        """生成趋势破坏的负样本 - 用于趋势对比损失"""
        B, C, T = x.shape
        x_corrupt = x.clone()
        
        for i in range(B):
            for var_idx in range(C):
                method = np.random.choice(self.trend_corruption_methods)
                x_corrupt[i, var_idx, :] = method(x[i, var_idx, :])
        
        return x_corrupt
    
    def corrupt_for_seasonality(self, x):
        """生成季节破坏的负样本 - 用于季节对比损失"""
        B, C, T = x.shape
        x_corrupt = x.clone()
        
        for i in range(B):
            for var_idx in range(C):
                method = np.random.choice(self.seasonal_corruption_methods)
                x_corrupt[i, var_idx, :] = method(x[i, var_idx, :])
        
        return x_corrupt
    
    def _add_linear_trend(self, x):
        """添加随机线性趋势"""
        T = x.size(0)
        x_neg = x.clone()
        
        # 自适应强度：基于序列标准差
        var_std = x.std()
        adaptive_strength = 0.05 + 0.1 * torch.sigmoid(var_std / 10.0)
        
        # 随机趋势强度
        trend_strength = torch.randn(1, device=x.device) * adaptive_strength
        # 对称时间点
        time_points = torch.linspace(-1, 1, T, device=x.device)
        # 应用趋势
        trend = trend_strength * time_points
        x_neg += trend
        
        return x_neg
    
    def _reverse_trend(self, x):
        """趋势反转"""
        T = x.size(0)
        x_neg = x.clone()
        
        linear_trend = torch.linspace(-1, 1, T, device=x.device)
        trend_strength = 0.1 * x.std()
        x_neg = x - 2 * linear_trend * trend_strength
        
        return x_neg
    
    def _phase_shift(self, x):
        """相位偏移"""
        T = x.size(0)
        x_neg = x.clone()
        
        # 设置默认偏移量
        if T >= 24:
            shift = torch.randint(2, T//4, (1,)).item()
        else:
            shift = torch.randint(1, T//2, (1,)).item()
        
        x_neg = torch.roll(x_neg, shifts=shift)
        return x_neg
    
    def _seasonal_scale(self, x):
        """季节性缩放"""
        x_neg = x.clone()
        scale = 0.7 + torch.rand(1).to(x.device) * 0.6
        x_neg = x * scale 
        return x_neg
    
    def _detrend_and_reshuffle(self, x):
        """去趋势并重排"""
        T = x.size(0)
        x_neg = x.clone()
        
        # 1. 去除线性趋势
        t = torch.arange(T, device=x.device, dtype=torch.float32)
        ## 计算线性回归参数
        t_mean = t.mean()
        x_mean = x.mean()
        numerator = ((t - t_mean) * (x - x_mean)).sum()
        denominator = ((t - t_mean) ** 2).sum()
        if denominator == 0: # 防止除零
            slope = torch.tensor(0.0, device=x.device)
        else:
            slope = numerator / denominator
        intercept = x_mean - slope * t_mean
        linear_trend = slope * t + intercept
        detrended = x - linear_trend
        
        # 2. 分段重排
        segment_size = max(4, T // 4)
        segments = []
        for i in range(0, T, segment_size):
            segment = detrended[i:i+segment_size]
            if len(segment) > 0:
                segments.append(segment)
        
        # 随机重排片段
        indices = torch.randperm(len(segments))
        shuffled_segments = [segments[i] for i in indices]
        
        # 3. 重新组合
        reshuffled = torch.cat(shuffled_segments)
        
        # 长度处理
        if len(reshuffled) < T:
            padding = torch.full((T - len(reshuffled),), reshuffled[-1], device=x.device)
            reshuffled = torch.cat([reshuffled, padding])
        elif len(reshuffled) > T:
            reshuffled = reshuffled[:T]
        
        # 重新添加趋势
        x_neg = reshuffled + linear_trend
        
        return x_neg
    
    def _periodic_noise(self, x):
        """周期性噪声"""
        T = x.size(0)
        x_neg = x.clone()
        # 1.检测有效周期
        effective_period = self._get_effective_period(T)
        # 2.生成周期性噪声
        t = torch.arange(T, device=x.device, dtype=torch.float32)
        # 3.生成基础周期噪声
        base_noise = torch.sin(2 * torch.pi * t / effective_period)
        # 4.生成谐波成分
        harmonic1 = 0.5 * torch.sin(4 * torch.pi * t / effective_period)
        harmonic2 = 0.25 * torch.sin(6 * torch.pi * t / effective_period)
        # 5.组合噪声
        periodic_noise = base_noise + harmonic1 + harmonic2
        # 6.调整噪声强度
        noise_strength = 0.1 * x.std()
        scaled_noise = periodic_noise * noise_strength 
        # 7.相位随机化
        phase_shift = torch.rand(1, device=x.device) * 2 * torch.pi
        phase_shifted_noise = torch.sin(2 * torch.pi * t / effective_period + phase_shift) * noise_strength
        # 8.组合噪声并应用
        combined_noise = 0.7 * scaled_noise + 0.3 * phase_shifted_noise
        x_neg += combined_noise
        # 9. 针对T刚好为有效周期长度的特殊处理，防止完整的周期性噪声被模型"学习"为某种固定模式
        if T == effective_period:
            # 通过轻微的时间扭曲添加额外的随机性来打破固定模式
            time_warp = 0.05 * torch.randn(1, device=x.device)  
            warped_t = t * (1 + time_warp)
            extra_noise = 0.2 * torch.sin(2 * torch.pi * warped_t / effective_period) * noise_strength
            x_neg += extra_noise
        
        return x_neg
    
    def _get_effective_period(self, T):
        """根据序列长度和频率自适应确定有效周期"""
        # 基于频率映射确定基本周期
        period_mapping = {
            'H': 24,        # 小时数据：24小时周期
            'T': 96,    # 15/30分钟数据：96个点（24/48小时）
            'D': 7         # 天数据：7天周期
        }
        base_period = period_mapping.get(self.freq, 24) # 默认使用小时频率
        effective_period = None
        # 如果序列长度不足以容纳完整周期，使用子周期
        if T < base_period:  # 寻找能整除T的最大因子作为周期
            # effective_period = self._find_divisor(T, base_period)
            for divisor in [base_period // 2, base_period // 3, base_period // 4]: # 优先尝试target_period的约数
                if divisor >= 2 and T >= divisor:
                    effective_period = divisor
                    break
            if effective_period==None: # 如果都不行，effective_period仍为 None,使用 T的约数
                for divisor in range(min(T, base_period), 1, -1):
                    if T % divisor == 0:
                        effective_period = divisor
                        break
                
            if effective_period==None:    # 最后保障：使用T的一半
                effective_period =  max(2, T // 2)
        else:
            # 使用完整周期，但确保不超过序列长度
            effective_period = min(base_period, T)
        
        # 确保周期至少为2（周期为1没有意义）
        return max(2, effective_period)
    


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
            # 组合多种轻微扰动
            ## scale
            scale = 0.9 + torch.rand(1).to(x.device) * 0.2
            x_var_scale= x[:,var_idx,:] * scale 
            ## shift
            shift = torch.randn(1).to(x.device) * 0.05
            x_var_shift = x_var_scale + shift
            ##low_freq_noise
            x_var_noise = self._add_low_frequency_noise(x_var_shift)       
            x_aug[:,var_idx,:] = x_var_noise
        
        # 关键：因果变量完全保持不变
        return x_aug
    
    def causal_negative_augment(self, x):
        """
        因果破坏的负增强 - 核心思想：专门针对因果变量(即自动识别的关键变量)进行破坏性干预
        """
        neg_samples = []
        
        # 策略1: 因果变量置换（最有效的策略）
        neg_samples.append(self._causal_variable_permutation(x))
        # 策略2: 因果变量结构化破坏
        neg_samples.append(self._causal_variable_structured_corruption(x))
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
                x_neg[:,var_idx,:] = x[perm_indices, var_idx, :]
        
        return x_neg
    
    def _add_low_frequency_noise(self, data):
        """添加"""
        B, T = data.shape
        device = data.device
        
        # 计算合适的基础噪声强度
        data_std = data.std(dim=1, keepdim=True)
        base_strength = data_std * 0.15  # 15%标准差作为基础强度
        
        # 生成低频噪声
        low_freq_noise = torch.zeros(B, T, device=device)
        
        for i in range(B):
            # 生成随机游走噪声
            white_noise = torch.randn(T, device=device) * base_strength[i] * 0.1
            random_walk = torch.cumsum(white_noise, dim=0)
            
            # 适度平滑
            kernel_size = max(3, T // 20)
            weights = torch.ones(kernel_size, device=device) / kernel_size
            
            # 使用正确的padding确保输出长度不变
            padding = (kernel_size - 1) // 2  
            
            smoothed = torch.nn.functional.conv1d(
                random_walk.unsqueeze(0).unsqueeze(0), 
                weights.unsqueeze(0).unsqueeze(0), 
                padding=padding
            ).squeeze()
            
            # 确保长度匹配
            if smoothed.size(0) > T:
                smoothed = smoothed[:T]
            elif smoothed.size(0) < T:
                # 如果长度不足，用零填充
                pad_size = T - smoothed.size(0)
                smoothed = torch.nn.functional.pad(smoothed, (0, pad_size))
            
            low_freq_noise[i] = smoothed
        
        return data + low_freq_noise
    
    
    def _causal_variable_structured_corruption(self, x):
        """因果变量结构化破坏 -时间反转或模式反转"""
        x_neg = x.clone()
        B, C, T = x.shape
        
        for var_idx in self.causal_vars:
            # 随机选择结构化破坏方式
            corruption_type = np.random.choice(['pattern_inversion','time_reverse'])
            
            if corruption_type =='time_reverse' :
                x_neg[:, var_idx, :] = torch.flip(x[:, var_idx, :], dims=[1])
            else:  # pattern_inversion
                # 模式反转：将序列围绕均值翻转
                series_mean = x[:, var_idx, :].mean(dim=1, keepdim=True)
                x_neg[:, var_idx, :] = 2 * series_mean - x[:, var_idx, :]
        
        return x_neg

    def _causal_relationship_disruption(self, x):
        """因果关系破坏 - 样本内交换因果变量"""
        x_neg = x.clone()
        B, C, T = x.shape
        
        if len(self.causal_vars) >= 2:
            # 随机选择两个因果变量交换它们的时序模式
            var1, var2 = np.random.choice(self.causal_vars, 2, replace=False)
            
            # 交换时序模式（保持各自的幅度特性）
            var1_data = x[:, var1,:].clone()
            var2_data = x[:, var2,:].clone()
            
            # 标准化后交换
            var1_mean, var1_std = var1_data.mean(), var1_data.std()
            var2_mean, var2_std = var2_data.mean(), var2_data.std()
            
            if var1_std > 1e-6 and var2_std > 1e-6:
                # 标准化交换后再还原
                var1_normalized = (var1_data - var1_mean) / var1_std
                var2_normalized = (var2_data - var2_mean) / var2_std
                
                x_neg[:, var1,:] = var2_normalized * var1_std + var1_mean
                x_neg[:, var2,:] = var1_normalized * var2_std + var2_mean
        
        return x_neg
    



class DynamicPeakDetectionAugmenter:
    """
    基于局部极值点的动态峰谷检测增强器
    专门针对负荷变量进行峰谷检测和增强
    """
    def __init__(self, max_augment_strength=0.15, noise_std_ratio=0.05,
                 min_peak_prominence_ratio=0.1, min_valley_prominence_ratio=0.08):
        self.max_augment_strength = max_augment_strength
        self.noise_std_ratio = noise_std_ratio
        self.min_peak_prominence_ratio = min_peak_prominence_ratio  # 波峰显著性阈值
        self.min_valley_prominence_ratio = min_valley_prominence_ratio  # 波谷显著性阈值
    
    def find_significant_extrema(self, sequence):
        """
        找到显著的局部极值点，考虑极值点的突出程度
        """
        T = sequence.size(0)
        
        # 检测局部极值点
        diff = torch.diff(sequence, prepend=sequence[0:1])
        sign_change = torch.diff(torch.sign(diff), prepend=torch.sign(diff[0:1]))
        
        # 局部极大值：符号由正变负
        local_max_mask = sign_change < 0
        # 局部极小值：符号由负变正  
        local_min_mask = sign_change > 0
        
        # 筛选显著的极值点
        significant_peaks = []
        significant_valleys = []
        
        sequence_mean = sequence.mean()
        sequence_std = sequence.std()
        
        # 检查每个局部极大值
        peak_indices = local_max_mask.nonzero().squeeze()
        if peak_indices.dim() > 0:  # 确保有极值点
            for idx in peak_indices:
                if idx == 0 or idx == T-1:  # 跳过端点
                    continue
                
                # 计算波峰的突出程度（与相邻极小值的差值）
                left_min = min(sequence[max(0, idx-5):idx])  # 左边5个点的最小值
                right_min = min(sequence[idx:min(T, idx+5)])  # 右边5个点的最小值
                prominence = sequence[idx] - max(left_min, right_min)
                
                # 只有突出程度足够大的波峰才被认为是显著的
                if prominence > self.min_peak_prominence_ratio * sequence_std:
                    significant_peaks.append(idx)
        
        # 检查每个局部极小值
        valley_indices = local_min_mask.nonzero().squeeze()
        if valley_indices.dim() > 0:
            for idx in valley_indices:
                if idx == 0 or idx == T-1:  # 跳过端点
                    continue
                
                # 计算波谷的突出程度
                left_max = max(sequence[max(0, idx-5):idx])  # 左边5个点的最大值
                right_max = max(sequence[idx:min(T, idx+5)])  # 右边5个点的最大值
                prominence = min(left_max, right_max) - sequence[idx]
                
                # 只有突出程度足够大的波谷才被认为是显著的
                if prominence > self.min_valley_prominence_ratio * sequence_std:
                    significant_valleys.append(idx)
        
        return significant_peaks, significant_valleys
    
    def dynamic_peak_augment(self, x, load_var_index):
        """
        基于局部极值点的动态峰谷增强
        """
        x_aug = x.clone()
        B, C, T = x.shape
        
        for i in range(B):
            sample_load = x[i, load_var_index, :]
            
            # 找到显著的波峰和波谷
            peak_indices, valley_indices = self.find_significant_extrema(sample_load)
            
            # 波峰增强：轻微衰减（模拟需求响应）
            if len(peak_indices) > 0:
                # 对每个波峰应用不同的随机缩放
                for peak_idx in peak_indices:
                    # 在波峰周围创建一个小的窗口（前后2个点）
                    start_idx = max(0, peak_idx - 2)
                    end_idx = min(T, peak_idx + 3)  # 包含peak_idx+2
                    
                    # 随机缩放因子，轻微衰减
                    peak_scale = 1.0 - torch.rand(1).to(x.device) * self.max_augment_strength
                    x_aug[i, :, start_idx:end_idx] = x_aug[i, :, start_idx:end_idx] * peak_scale
            
            # 波谷增强：添加轻微噪声（模拟基础负荷波动）
            if len(valley_indices) > 0:
                for valley_idx in valley_indices:
                    # 在波谷周围创建窗口
                    start_idx = max(0, valley_idx - 1)
                    end_idx = min(T, valley_idx + 2)  # 包含valley_idx+1
                    
                    # 基于局部数据的标准差设置噪声强度
                    valley_data = x_aug[i, :, start_idx:end_idx]
                    local_std = valley_data.std()
                    noise_std = torch.rand(1).to(x.device) * local_std * self.noise_std_ratio
                    
                    # 添加高斯噪声
                    noise = torch.randn_like(valley_data) * noise_std
                    x_aug[i, :, start_idx:end_idx] = valley_data + noise
        
        return x_aug


class TemporalAugmenter:
    """
    通用时序增强器。
    """
    def __init__(self):
        pass
   
    def _fluctuation_pattern_augment(self, x):
        """
        波动强度自适应的相关性噪声增强。
        """
        B, C, T = x.shape
        x_aug = x.clone()
        for i in range(B): 
            # 计算当前样本每个变量的波动性
            volatility = torch.std(x_aug[i], dim=1) # [C]
            
            # 计算分位数
            q1 = torch.quantile(volatility, 0.25)  # 25%分位数
            q3 = torch.quantile(volatility, 0.75)  # 75%分位数
            
            # 动态增强阈值：只对波动性高于Q1的变量增强
            threshold = q1
            for var_idx in range(C):
                if volatility[var_idx] > threshold:
                    # 基于波动性设置增强强度
                    base_strength = 0.05  # 基础强度
                    if volatility[var_idx] > q3:  # 高波动性变量：较强增强
                        relative_volatility = volatility[var_idx] / q3
                        aug_strength = base_strength * min(2.0, relative_volatility)  # 限制最大2倍
                    else: # 中等波动性变量：固定轻度增强
                        aug_strength = base_strength  
                    
                    # 生成相关性噪声（低通滤波）
                    white_noise = torch.randn(T).to(x.device)
                    # 简单移动平均平滑
                    kernel_size = 3
                    weights = torch.ones(kernel_size).to(x.device) / kernel_size
                    smoothed_noise = torch.conv1d(
                        white_noise.unsqueeze(0).unsqueeze(0), 
                        weights.unsqueeze(0).unsqueeze(0), 
                        padding=kernel_size//2).squeeze()
                    x_aug[i, var_idx, :] = x_aug[i, var_idx, :] + smoothed_noise * aug_strength
        
        return x_aug

class DomainAugmentationFramework:
    """
    基于领域知识的数据增强框架
    """
    def __init__(self, args):
        self.target_index = args.pretrain_target_idx
        self.importance_analyzer = VariableImportanceAnalyzer(args.pretrain_target_idx)
        self.causal_augmenter = None
        self.peak_augmenter = DynamicPeakDetectionAugmenter()
        self.temporal_augmenter = TemporalAugmenter()
        self.trend_seasonal_augmenter = TrendSeasonalAugmenter(args.pretrain_freq)

        
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
        
        # 3. 负增强：生成趋势或季节性单成分破坏的负样本
        trend_negative_list = [self.trend_seasonal_augmenter.corrupt_for_trend(neg_x)
                       for neg_x in negative_x_list]
        seasonal_negative_list = [self.trend_seasonal_augmenter.corrupt_for_seasonality(neg_x)
                      for neg_x in negative_x_list]
        
        # 4.通用增强：丰富正负样本
        ## 负样本处理
        t_negative_list = [self.temporal_augmenter._fluctuation_pattern_augment(x)
                           for x in trend_negative_list]
        s_negative_list = [self.temporal_augmenter._fluctuation_pattern_augment(x)
                           for x in seasonal_negative_list]
        ## 正样本处理
        positive_x = self.temporal_augmenter._fluctuation_pattern_augment(positive_x)
        
        # 可视化
        if plot_augment_flag and plot_dir:
            self.visualize_augmentations(
                batch_x, positive_x, negative_x_list,t_negative_list,s_negative_list,
                self.causal_vars, plot_dir, plot_augment_flag, True)

        return batch_x, positive_x, negative_x_list,t_negative_list,s_negative_list
        
    def visualize_augmentations(self, original, positive,negatives, t_negatives,s_negatives, causal_vars,
                           plot_dir, plot_augment_flag=True, plot_augment=True):
        """
        可视化原始数据和增强数据
        original: [B, T, C] 原始数据
        positive: [B, T, C] 正样本
        negatives: list of [B, T, C] without trend/seasonal corruption 未经趋势性/季节性成分破坏的负样本列表
        t_negatives: list of [B, T, C] for trend 趋势性成分被破坏的负样本列表
        s_negatives: list of [B, T, C] for trend 季节性成分被破坏的负样本列表
        causal_vars: 因果变量索引列表
        """
        if not plot_augment_flag or not plot_augment:
            return
        
        batch_size, num_features,seq_len = original.shape
        negative_num = len(negatives)
        
        # 只可视化第一个样本
        sample_idx = 0
        
        for i in range(num_features):
            # 确定变量类型用于标签
            var_type = "causal" if i in causal_vars else "non-causal"
            
            for j in range(negative_num):
                plt.figure(figsize=(15, 5))
                
                # 提取数据 - 注意维度是 [B, C, T]
                orig_data = original[sample_idx,i, :].detach().cpu().numpy()
                pos_data = positive[sample_idx, i, :].detach().cpu().numpy()
                neg_data = negatives[j][sample_idx, i, :].detach().cpu().numpy()
                t_neg_data = t_negatives[j][sample_idx, i, :].detach().cpu().numpy()
                s_neg_data = s_negatives[j][sample_idx, i, :].detach().cpu().numpy()

                # 绘制五条线
                plt.plot(orig_data, label='original', linestyle='-', marker='o', markersize=3)
                plt.plot(pos_data, label='positive augment', linestyle='--', marker='x', markersize=3)
                plt.plot(neg_data, label=f'negative augment {j+1}', linestyle=':', marker='s', markersize=3)
                plt.plot(t_neg_data, label=f'trend negative augment {j+1}', linestyle=':', marker='*', markersize=3)
                plt.plot(s_neg_data, label=f'seasonal negative augment {j+1}', linestyle=':', marker='>', markersize=3)

                plt.title(f'{var_type} variable{i} - negative augmentation {j+1}')
                plt.xlabel('timestamp')
                plt.ylabel('value')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # 保存图片
                os.makedirs(plot_dir, exist_ok=True)
                plt.savefig(os.path.join(plot_dir, f'var_{i}_causal_{i in causal_vars}_neg_{j+1}.png'), 
                        dpi=150, bbox_inches='tight')
                plt.close()

