import numpy as np
from scipy import signal
from scipy.stats import pearsonr


"""
=====原始尺度基础评估指标=====
"""

def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))

"""
尺度归一化的基础指标
"""
def normalized_MAE(pred, true):
    """
    归一化平均绝对误差
    除以真实值的均值，消除尺度影响
    """
    mask_zero = true != 0
    mean_true = np.mean(np.abs(true[mask_zero]))
    if mean_true == 0:
        return np.nan
    return MAE(pred, true) / mean_true

def normalized_MSE(pred, true):
    """
    归一化均方误差
    除以真实值平方的均值
    """
    mask_zero = true != 0
    mean_square_true = np.mean(true[mask_zero] ** 2)
    if mean_square_true == 0:
        return np.nan
    return MSE(pred, true) / mean_square_true


def normalized_RMSE(pred, true):
    """
    归一化均方根误差
    """
    return np.sqrt(normalized_MSE(pred, true))

def CV_RMSE(pred, true):
    """
    变异系数均方根误差 (Coefficient of Variation of RMSE)
    常用于建筑能耗预测评估
    """
    mask_zero = true != 0
    mean_true = np.mean(true[mask_zero])
    if mean_true == 0:
        return np.nan
    return RMSE(pred, true) / mean_true  





"""
=====百分比基础评估指标=====
"""
def MAPE(pred, true):
    """鲁棒的平均绝对百分比误差与平均平方百分比误差"""
    # 使用相对阈值进行掩码
    mask_zero = true != 0
    
    if not mask_zero.any():
        return np.nan, np.nan
    
    mean_true = np.mean(np.abs(true[mask_zero]))
    thresh = max(mean_true * 0.01, 1e-3)
    mask = np.abs(true) > thresh
    mape = np.mean(np.abs(pred[mask] - true[mask]) / np.abs(true[mask]))
    
    return mape    


"""
=============尺度无关基础指标================
"""
def MASE(pred, true, seasonal_period=1):
    """
    平均绝对标度误差
    公式：MASE = MAE(pred, true) / MAE(naive_forecast, true)
    特点：与尺度无关，便于比较不同尺度的时间序列
    """
    # naive预测：使用前一个周期相同位置的值作为预测
    naive_pred = np.roll(true, seasonal_period)
    naive_pred[:seasonal_period] = true[:seasonal_period]  # 前seasonal_period个点用自身填充
    
    # 计算MAE
    model_mae = MAE(pred, true)
    naive_mae = MAE(naive_pred[seasonal_period:], true[seasonal_period:])
    
    if naive_mae == 0:
        return np.nan
    return model_mae / naive_mae

def R_squared(pred, true):
    """
    决定系数 R²
    表示模型解释的方差比例，与尺度无关
    """
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - np.mean(true)) ** 2)
    if ss_tot == 0:
        return np.nan
    return 1 - (ss_res / ss_tot)


"""
=========峰值指标========
"""
def peak_error(pred, true):
    """
    峰值误差 - 计算每个序列的峰值预测误差
    假设输入形状: [batch_size, seq_len, 1]
    返回：绝对峰值误差、相对峰值误差、归一化绝对峰值误差
    """
    # 确保输入形状正确
    if pred.ndim != 3 or pred.shape[-1] != 1 or true.shape[-1] != 1:
        raise ValueError(f"输入形状应为 [batch_size, seq_len, 1]，但得到 pred: {pred.shape}, true: {true.shape}")
    
    batch_size, seq_len, _ = pred.shape
    abs_errors = []
    rel_errors = []
    normalized_abs_errors = []
    
    # 对每个batch单独处理
    for i in range(batch_size):
        # ===================== 初步峰值检测 =====================
        true_seq = true[i, :, 0].flatten()  # [seq_len]
        pred_seq = pred[i, :, 0].flatten()  # [seq_len]
        
        if (true_seq > 0.1 ).any(): 
            percentile = np.percentile(true_seq[true_seq > 0.1], 90)
        else: # true_seq 全为 极小值0
            continue
        
        # 检测真实序列的峰值
        true_peaks_idx, true_peaks_props = signal.find_peaks(
            true_seq, 
            height = percentile,
            prominence=np.std(true_seq) * 0.5  # 增加prominence避免检测噪声峰值
        )
        
        # 检测预测序列的峰值
        pred_peaks_idx, pred_peaks_props = signal.find_peaks(
            pred_seq,
            height=percentile,
            prominence=np.std(pred_seq) * 0.5
        )

        # ===================== 真实峰值过滤 =====================
        # 步骤1：对真实峰值按「下标相邻≤4」分组
        filtered_true_peaks = []
        if len(true_peaks_idx) > 0:
            peak_groups = [] # 初始化第一个分组
            current_group = [true_peaks_idx[0]]
            
            for idx in true_peaks_idx[1:]: # 遍历剩余峰值，按间距分组
                if idx - current_group[-1] <= 4:  # 相邻下标差≤4则归为同一组
                    current_group.append(idx)
                else:  # 超过4则新建分组
                    peak_groups.append(current_group)
                    current_group = [idx]
            peak_groups.append(current_group)  # 加入最后一个分组
            
            # 步骤2：每组保留值最大的峰值下标
            for group in peak_groups:
                group_peak_values = [true_seq[idx] for idx in group] # 提取组内每个下标对应的峰值大小
                max_val_pos = np.argmax(group_peak_values) # 找到组内最大值对应的索引
                filtered_true_peaks.append(group[max_val_pos]) # 保留最大值对应的下标
        
        # ===================== 误差计算（基于筛选后的峰值） =====================
        # 如果筛选后有真实峰值，且预测有峰值，则计算误差
        if len(filtered_true_peaks) > 0 and len(pred_peaks_idx) > 0:
            # 为每个真实峰值找到最近的预测峰值
            for true_idx in filtered_true_peaks:
                true_peak_value = true_seq[true_idx]
                
                # 找到时间上最接近的预测峰值
                closest_pred_idx = pred_peaks_idx[np.argmin(np.abs(pred_peaks_idx - true_idx))]
                pred_peak_value = pred_seq[closest_pred_idx]
                
                # 计算误差
                abs_error = np.abs(pred_peak_value - true_peak_value)
                #if abs_error>10:
                #    print(pred_peak_value)
                #    print(true_peak_value)
                #    print(pred_peaks_idx)
                #    print(true_peaks_idx)
                #    print(closest_pred_idx)
                    

                rel_error = abs_error / (np.abs(true_peak_value) + 1e-8)
                #if rel_error>100:
                #    print(true_peak_value)
                #    print(pred_peak_value)
                #    print(pred_peaks_idx)
                #    print(true_peaks_idx)
                #    print(closest_pred_idx)
                #    print(true_peak_value)
                
                # 归一化绝对误差：除以整个序列的范围
                seq_range = np.max(true_seq) - np.min(true_seq)
                #if seq_range<=1:
                #    print(true_seq)
                #    print(np.max(true_seq))
                #    print(np.min(true_seq))
                if seq_range > 1:
                    normalized_abs_error = abs_error / seq_range
                    normalized_abs_errors.append(normalized_abs_error)
                
                abs_errors.append(abs_error)
                rel_errors.append(rel_error)
                
    
    # 返回平均值
    if abs_errors:
        return (
            np.mean(abs_errors), 
            np.mean(rel_errors), 
            np.mean(normalized_abs_errors)
        )
    else:
        return np.nan, np.nan, np.nan

def peak_time_shift(pred, true, time_step=1):
    """
    峰值时间偏移 - 计算峰值出现时间的偏移
    假设输入形状: [batch_size, seq_len, 1]
    time_step: 每个时间步的实际时间（小时/分钟等）
    """
    # 确保输入形状正确
    if pred.ndim != 3 or pred.shape[-1] != 1 or true.shape[-1] != 1:
        raise ValueError(f"输入形状应为 [batch_size, seq_len, 1]，但得到 pred: {pred.shape}, true: {true.shape}")
    
    batch_size, seq_len, _ = pred.shape
    time_shifts = []
    
    # 对每个batch单独处理
    for i in range(batch_size):
        true_seq = true[i, :, 0].flatten()  # [seq_len]
        pred_seq = pred[i, :, 0].flatten()  # [seq_len]
        
        # 检测真实序列的峰值
        true_peaks_idx, true_peaks_props = signal.find_peaks(
            true_seq, 
            height=np.percentile(true_seq, 75),
            prominence=np.std(true_seq) * 0.5
        )
        
        # 检测预测序列的峰值
        pred_peaks_idx, pred_peaks_props = signal.find_peaks(
            pred_seq,
            height=np.percentile(pred_seq, 75),
            prominence=np.std(pred_seq) * 0.5
        )
        
        # 如果至少有一个峰值，则进行匹配
        if len(true_peaks_idx) > 0 and len(pred_peaks_idx) > 0:
            # 为每个真实峰值找到最近的预测峰值
            for true_idx in true_peaks_idx:
                closest_pred_idx = pred_peaks_idx[np.argmin(np.abs(pred_peaks_idx - true_idx))]
                time_shift = np.abs(closest_pred_idx - true_idx) * time_step
                time_shifts.append(time_shift)
    
    # 返回平均值
    if time_shifts:
        return np.mean(time_shifts)
    else:
        return np.nan

"""
=============相关性指标=============
"""
def correlation_coefficient(pred, true):
    """相关系数 - 衡量预测与真实值之间的线性相关程度"""
    if pred.ndim == 3:  # 多变量
        corrs = []
        for var_idx in range(pred.shape[-1]):
            true_var = true[:, :, var_idx].flatten()
            pred_var = pred[:, :, var_idx].flatten()
            
            # 使用scipy的pearsonr
            corr, _ = pearsonr(true_var, pred_var)
            if not np.isnan(corr):
                corrs.append(corr)
        
        if corrs:
            return np.mean(corrs)
        else:
            return np.nan
    else:
        true_flat = true.flatten()
        pred_flat = pred.flatten()
        corr, _ = pearsonr(true_flat, pred_flat)
        return corr



def metric(pred, true,seasonal_period=24,time_step=1):
    """
    综合评估指标计算
    seasonal_period: 季节周期，用于MASE计算
    time_step: 时间步长（小时/分钟），用于峰值时间偏移计算
    """
    # 原始尺度基础指标
    mae = MAE(pred, true)
    rmse = RMSE(pred, true)

    # 归一化尺度基础指标
    nmae = normalized_MAE(pred,true)
    nrmse = normalized_RMSE(pred,true) 
    cv_rmse = CV_RMSE(pred,true)

    # 百分比基础指标(天然尺度无关指标)
    mape = MAPE(pred, true)
    
    # 其它天然尺度无关指标
    mase_val = MASE(pred, true, seasonal_period)
    R2 = R_squared(pred,true)
    
    # 峰值相关指标
    peak_abs_error, peak_rel_error,normalized_peak_abs_error = peak_error(pred, true)
    peak_time_shift_val = peak_time_shift(pred, true, time_step)

    # 相关性指标
    corr_coef = correlation_coefficient(pred, true)
    
    # 返回所有指标
    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'NMAE':nmae,
        'NRMSE':nrmse,
        'CV_RMSE':cv_rmse,
        'MAPE': mape,
        'MASE': mase_val,
        'R2':R2,
        'Peak_Abs_Error': peak_abs_error,
        'Peak_Rel_Error': peak_rel_error,
        'Normalized_Peak_Abs_Error':normalized_peak_abs_error,
        'Peak_Time_Shift': peak_time_shift_val,
        'Correlation': corr_coef
    }
    
    return metrics
