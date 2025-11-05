import os
import pandas as pd
import numpy as np
import math
from torch.utils.data import Dataset
from utils.timefeatures import time_features
from sklearn.preprocessing import StandardScaler


class CustomDataset(Dataset):
    def __init__(self, args, flag='train',  data_path=None, pretrain_stage=False):
        self.args = args
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.stride = args.stride
        
        assert flag in ['train', 'test', 'val']
        self.flag = flag
        self.set_type = {'train': 0, 'val': 1, 'test': 2}[flag]
        self.scale = args.scale
        self.freq = args.freq  # 数据集采样频率参数中的频率
        self.root_path = args.root_path
        self.data_path = data_path if data_path is not None else args.data_path
        self.pretrain_stage = pretrain_stage
        self.targetidx = args.target_idx  # 直接使用列下标

        self.__read_data__()

        # 动态设置模型输入维度
        args.input_size = self.data_x.shape[1]

    def __read_data__(self):
        """
        读取和预处理数据，支持不同模式。根据配置，该方法将执行以下操作：
        1. 读取CSV文件。
        2. 解析多种格式的'date'列的时间戳。
        3. 对非日期列进行数据清洗（根据需要填充缺失值、进行标准化以及进行数据增强）。
        4. 根据需要分割数据集，在single模式或者one2many模式的非预训练阶段，将数据集分割为训练、验证和测试集。
        5. 使用'time_feature'函数对'date'列进行时间编码。
        注意：该方法不返回任何值，但它修改了实例的属性以包含预处理后的数据。
        """
        
        self.scaler = StandardScaler()
        
        # 1. 读取数据
        self.df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        print(f"Loaded dataset: {self.data_path}, shape: {self.df_raw.shape}")
        
        # 2. 日期列处理
        if 'date' in self.df_raw.columns:
            # 先查看原始日期数据的样本和类型
            print("Date column sample (first 5):", self.df_raw['date'].head().tolist())
            print("Date column dtype:", self.df_raw['date'].dtype)
            
            # 对于电工数学建模数据集，日期格式是 YYYYMMDD 整数
            if self.df_raw['date'].dtype in [np.int64, np.float64, int, float]:
                print("Detected numeric date format (YYYYMMDD), converting...")
                # 将整数转换为字符串，然后解析为日期
                date_strings = self.df_raw['date'].astype(str)
                self.df_raw['date'] = pd.to_datetime(date_strings, format='%Y%m%d', errors='coerce')
            else:
                # 对于字符串日期，尝试多种格式（并在失败时回退到 pandas 的自动解析以支持非零填充的日期如 '2006/1/1 0:00'）
                date_formats = [
                    '%Y%m%d',  # 20120105
                    '%Y/%m/%d %H:%M',  # 2006/8/11 18:30 or 2006/1/1 0:00 (may be padded or not)
                    '%Y-%m-%d %H:%M:%S',  # 2019-01-01 01:00:00
                    '%Y-%m-%d',  # 2019-01-01
                ]
                
                parsed_successfully = False
                for fmt in date_formats:
                    try:
                        # 先尝试使用指定格式解析
                        parsed = pd.to_datetime(self.df_raw['date'], format=fmt, errors='coerce')
                        
                        # 检查解析结果
                        null_count = parsed.isnull().sum()
                        if null_count == 0:
                            self.df_raw['date'] = parsed
                            print(f"Successfully parsed dates with format: {fmt}")
                            parsed_successfully = True
                            break
                        else:
                            print(f"Format {fmt} failed: {null_count} null values")
                            
                    except Exception as e:
                        print(f"Format {fmt} error: {e}")
                        continue
                
                if not parsed_successfully:
                    # 回退：让 pandas / dateutil 自动解析（能够处理像 '2006/1/1 0:00' 这种没有零填充的时间）
                    try:
                        parsed = pd.to_datetime(self.df_raw['date'], errors='coerce', infer_datetime_format=True)
                        null_count = parsed.isnull().sum()
                        if null_count == 0:
                            self.df_raw['date'] = parsed
                            print("Successfully parsed dates with pandas' automatic parser")
                            parsed_successfully = True
                        else:
                            print(f"Automatic parsing produced {null_count} null values")
                    except Exception as e:
                        print(f"Automatic parsing error: {e}")
                
                if not parsed_successfully:
                    print("Warning: Could not parse date column with any format, using relative time")
                    self.df_raw['date'] = pd.date_range(start='2000-01-01', periods=len(self.df_raw), freq=self.freq)
        else:
            # 如果没有日期列，创建相对时间
            print("No date column found, creating relative time")
            self.df_raw['date'] = pd.date_range(start='2000-01-01', periods=len(self.df_raw), freq=self.freq)

        # 3. 数据清洗(仅仅针对非日期列)
        cols_data = [col for col in self.df_raw.columns if col != 'date']
        df_data = self.df_raw[cols_data]
        
        # 处理缺失值和异常值
        df_data.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_data.fillna(df_data.mean(), inplace=True)
        
        # 4. 根据模式确定数据边界
        if self.pretrain_stage and self.args.pretrain_mode == 'one2many':
            # One2Many 预训练阶段：使用完整数据集
            border1, border2 = 0, len(df_data)
            print(f"One2Many Pretrain - Using full dataset: {border1} to {border2}")
        else:
            # Single 模式 或 One2Many 微调阶段：按比例划分
            num_train = int(len(df_data) * 0.7)
            num_test = int(len(df_data) * 0.2)
            num_vali = len(df_data) - num_train - num_test
            
            border1s = [0, num_train - self.seq_len, len(df_data) - num_test - self.seq_len]
            border2s = [num_train, num_train + num_vali, len(df_data)]
            
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]
            print(f"{self.flag} set - Using data: {border1} to {border2}")
        
        # 5. 数据标准化
        if self.scale:
            if self.pretrain_stage and self.args.pretrain_mode == 'one2many':
                self.scaler.fit(df_data.values)
            else:
                train_border1 = 0 if (self.pretrain_stage and self.args.pretrain_mode == 'one2many') else border1s[0]
                train_border2 = len(df_data) if (self.pretrain_stage and self.args.pretrain_mode == 'one2many') else border2s[0]
                train_data = df_data[train_border1:train_border2]
                self.scaler.fit(train_data.values)
            
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        
        # 转换为 float32 避免类型问题
        data = data.astype(np.float32)
        
        # 6. 设置数据
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2, self.targetidx]
        
        # 7. 使用现成的时间编码工具
        dates = self.df_raw['date'].iloc[border1:border2]
        self.data_stamp = time_features(dates, freq=self.freq).transpose(1, 0)  # 转置为 [seq_len, time_features]
        
        print(f"Final - data_x: {self.data_x.shape}, data_y: {self.data_y.shape}, data_stamp: {self.data_stamp.shape}")

    def __getitem__(self, index):
        """本质上是在通过滑动窗口创建样本并构建其标签"""
        s_begin = index * self.stride
        s_end = s_begin + self.seq_len
        r_begin = s_end 
        r_end = r_begin + self.pred_len
        
        # 边界检查
        if s_end > len(self.data_x) or r_end > len(self.data_y) or s_end > len(self.data_stamp) or r_end > len(self.data_stamp):
            raise IndexError(f"Index out of bounds: index={index}")
        
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        # 严格的形状检查和修复
        # 确保所有数组都是2D的
        if seq_x.ndim == 1:
            seq_x = seq_x.reshape(-1, 1)
        if seq_y.ndim == 0:  # 标量情况
            seq_y = np.array([seq_y]).reshape(-1, 1)
        elif seq_y.ndim == 1:
            seq_y = seq_y.reshape(-1, 1)
        if seq_x_mark.ndim == 1:
            seq_x_mark = seq_x_mark.reshape(-1, 1)
        if seq_y_mark.ndim == 1:
            seq_y_mark = seq_y_mark.reshape(-1, 1)

        seq_x = seq_x.astype(np.float32)
        seq_y = seq_y.astype(np.float32)
        seq_x_mark = seq_x_mark.astype(np.float32)
        seq_y_mark = seq_y_mark.astype(np.float32)
        
        return seq_x, seq_x_mark, seq_y, seq_y_mark
    
    def __len__(self):
        total_length = (len(self.data_x) - self.seq_len - self.pred_len) // self.stride + 1
        return max(0, total_length)
    
    
    def get_max_iterations(self, batch_size):
        return math.ceil(len(self) / batch_size)
    
    def extract_features(self):
        from scipy.fftpack import fft
        features = []
        for index in range(len(self)):
            seq_x, seq_y, seq_x_mark, seq_y_mark = self.__getitem__(index)
            mean = np.mean(seq_x, axis=0)
            std = np.std(seq_x, axis=0)
            skew = np.mean((seq_x - mean)**3, axis=0) / std**3
            kurt = np.mean((seq_x - mean)**4, axis=0) / std**4 - 3
            fft_features = np.abs(fft(seq_x, axis=0))[:self.seq_len // 2]
            fft_mean = np.mean(fft_features, axis=0)
            
            feature_vector = np.concatenate([mean, std, skew, kurt, fft_mean])
            features.append(feature_vector)
        return np.array(features)
   



