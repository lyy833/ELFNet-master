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
            # 组合多种轻微扰动
            ## scale
            scale = 0.9 + torch.rand(1).to(x.device) * 0.2
            x_var_scale= x[:,var_idx,:] * scale 
            ## shift
            shift = torch.randn(1).to(x.device) * 0.01
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
        """因果变量结构化破坏 - 使用趋势反转和相位偏移替代随机噪声"""
        x_neg = x.clone()
        B, C, T = x.shape
        
        for var_idx in self.causal_vars:
            # 随机选择结构化破坏方式
            corruption_type = np.random.choice(['trend_reverse', 'phase_shift', 'pattern_inversion','time_reverse'])
            
            if corruption_type == 'trend_reverse':
                # 趋势反转：对序列进行线性变换使其趋势反转
                ## 生成一个从-1到1的等差数列，长度为T（时间步长），然后调整形状为(1, T)
                linear_trend = torch.linspace(-1, 1, T, device=x.device).reshape(1, -1)
                ## 计算所有样本下标为var_idx的变量的标准差，然后乘以0.1,trend_strength的形状为(B, 1)
                trend_strength = 0.1 * x[:,var_idx,:].std(dim=1, keepdim=True)
                ## 这里linear_trend的形状是(1, T)，trend_strength的形状是(B, 1)，两者相乘会广播为(B, T)
                ## trend[t] * strength 提取了一个线性趋势
                ## 减去其两倍相当于实现趋势反转
                x_neg[:,var_idx,:] = x[:,var_idx, :] - 2 * linear_trend * trend_strength
                
            elif corruption_type == 'phase_shift':
                # 相位偏移：将序列在时间轴上平移
                shift_amount = torch.randint(T//4, T//2, (1,)).item()
                x_neg[:,var_idx,:] = torch.roll(x_neg[:,var_idx,:], shifts=shift_amount, dims=1)
            elif corruption_type =='time_reverse' :
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
    动态峰谷检测增强器,专门针对负荷变量进行峰谷检测,
    使用分位数阈值识别高负荷和低负荷时段,基于电力系统实际的峰谷用电模式进行增强
    """
    def __init__(self, peak_percentile=0.85, off_peak_percentile=0.15, 
                 min_std_ratio=1.0, min_range_ratio=0.3,max_augment_strength=0.15):
        self.peak_percentile = peak_percentile # 波峰分位数
        self.off_peak_percentile = off_peak_percentile # 波谷分位数
        self.min_std_ratio = min_std_ratio  # 峰谷最小比例阈值
        self.min_range_ratio = min_range_ratio
        self.max_augment_strength = max_augment_strength # 最大增强强度
    
    def dynamic_peak_augment(self, x, load_var_index):
        """
        动态峰谷检测和增强。
        - 波峰衰减：模拟需求响应、能效措施、分布式发电
        - 波谷加噪：模拟基础负荷波动、小用户随机行为、测量噪声
        """
        x_aug = x.clone()
        B, C, T = x.shape
        
        load_data = x[:, load_var_index,:]  # [B, T]
        
        for i in range(B): # 依次处理每个样本
            sample_load = load_data[i] # [T]
            
            # 计算峰谷阈值
            peak_threshold = torch.quantile(sample_load, self.peak_percentile) # 默认sample_load升序排列后T*85%位置对应元素的值
            off_peak_threshold = torch.quantile(sample_load, self.off_peak_percentile) # # 默认sample_load升序排列后T*15%位置对应元素的值
            
            # 峰谷差异显著性检查,结合多种指标进行检查
            data_std = sample_load.std()
            data_range = sample_load.max() - sample_load.min()
            peak_valley_gap = peak_threshold - off_peak_threshold
            ## 使用标准差和范围的综合指标
            std_ratio = peak_valley_gap / (data_std + 1e-6)
            range_ratio = peak_valley_gap / (data_range + 1e-6)
            ## 任一指标满足即可
            if std_ratio < self.min_std_ratio and range_ratio < self.min_range_ratio:
                continue
            
            peak_mask = sample_load > peak_threshold
            off_peak_mask = sample_load < off_peak_threshold
            
            # 峰时段增强
            if peak_mask.any():
                # max_augment_strength用于最大增强强度限制，防止过度增强导致数据失真
                peak_scale = 1.0 - torch.rand(1).to(x.device) * self.max_augment_strength
                x_aug[i,: ,peak_mask] = x_aug[i,:,peak_mask] * peak_scale
            
            # 谷时段增强
            if off_peak_mask.any():
                # 基于谷时段数据的标准差设置噪声强度
                off_peak_data = x_aug[i, :, off_peak_mask]
                data_std = off_peak_data.std()
                noise_std = torch.rand(1).to(x.device) * data_std * 0.1  # 10%的标准差
                noise = torch.randn_like(off_peak_data) * noise_std
                x_aug[i, :, off_peak_mask] = off_peak_data + noise
        
        return x_aug


class TemporalAugmenter:
    """
    通用时序局部模式增强器。
    """
    def __init__(self):
        pass
    
    def _local_extremum_augment(self, x):
        """
        基于局部极值点的增强
        """
        B, C, T = x.shape
        x_aug = x.clone()
        for i in range(B): # 依次处理每个样本
            for var_idx in range(C): # 每个样本中依次处理每个变量
                var_series = x_aug[i, var_idx,:]
                
                # 检测局部极值
                diff = torch.diff(var_series, prepend=var_series[0:1])
                threshold = 1e-4 # 设置一个阈值，忽略微小变化
                significant_changes = torch.abs(diff) > threshold
                diff_sign = torch.sign(diff)
                diff_sign[~significant_changes] = 0  # 微小变化视为平稳

                diff_sign_change = torch.diff(diff_sign, prepend=diff_sign[0:1])
                # 极值点：符号变化非零的位置
                extremum_mask = diff_sign_change != 0
                
                if extremum_mask.any():
                    # 对极值点进行轻微扰动
                    extremum_scale = 0.9 + torch.rand(1).to(x.device) * 0.2 # 缩放因子范围为 [0.9，1.1]
                    x_aug[i,var_idx,extremum_mask] = x_aug[i,var_idx,extremum_mask] * extremum_scale
        
        return x_aug
    
    def _fluctuation_pattern_augment(self, x):
        """
        基于波动模式的增强 - 使用分位数范围
        """
        B, C, T = x.shape
        x_aug = x.clone()
        for i in range(B):
            volatility = torch.std(x_aug[i], dim=1)
            
            # 计算分位数
            q1 = torch.quantile(volatility, 0.25)  # 25%分位数
            q3 = torch.quantile(volatility, 0.75)  # 75%分位数
            
            # 动态阈值：只对波动性高于Q1的变量增强
            # 对高于Q3的变量使用更强的增强
            threshold = q1
            
            for var_idx in range(C):
                if volatility[var_idx] > threshold:
                    # 确定增强强度
                    if volatility[var_idx] > q3:
                        # 高波动性变量：较强增强
                        aug_strength = 0.1 + torch.rand(1).to(x_aug.device) * 0.1  # 0.1-0.2
                    else:
                        # 中等波动性变量：适度增强
                        aug_strength = 0.05 + torch.rand(1).to(x_aug.device) * 0.05  # 0.05-0.1
                    
                    noise = torch.randn(T).to(x_aug.device) * aug_strength
                    x_aug[i, var_idx, :] = x_aug[i, var_idx, :] + noise
        
        return x_aug

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
        
        # 3. 正增强：通用时序局部模式增强
        positive_x = self.temporal_augmenter._local_extremum_augment(positive_x)
        
        # 对负样本也应用通用时序局部模式增强以增加多样性
        enhanced_negative_list = []
        for neg_x in negative_x_list:
            neg_enhanced = self.temporal_augmenter._fluctuation_pattern_augment(neg_x)
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
                
                # 绘制三条线
                plt.plot(orig_data, label='original', linestyle='-', marker='o', markersize=3)
                plt.plot(pos_data, label='positive augment', linestyle='--', marker='x', markersize=3)
                plt.plot(neg_data, label=f'negative augment {j+1}', linestyle=':', marker='s', markersize=3)
                
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

