import os
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

def plot_raw_df(df, x, y_columns, title='', xlabel='', ylabel='Value', dpi=300):	
    plt.figure(figsize=(6.93,5), dpi=dpi)	
    for column in y_columns:
        plt.plot(df[x], df[column], label=column)	
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.legend()
    plt.show()

def plot_resample_df(df, column, yearly=True, monthly=True, weekly=True, daily=True):
    
    # 设置全局字体大小
    plt.rcParams.update({'font.size': 5})  # 将10替换为您希望的字体大小
    plt.figure(figsize=(6.93,5),dpi=300)
    
    plot_count = sum([yearly, monthly, weekly, daily])
    current_plot = 1
    
    if yearly:
        yearly_resample = df[column].resample('Y').mean()
        plt.subplot(plot_count, 1, current_plot)
        yearly_resample.plot()
        plt.xlabel('Yearly',fontsize=5)
        plt.ylabel('',fontsize=5)
        plt.yticks([])  # 隐藏y轴刻度值
        plt.tick_params(axis='x', which='both', labelbottom=False)  # Hide x-axis labels
        current_plot += 1

    if monthly:
        monthly_resample = df[column].resample('M').mean()
        plt.subplot(plot_count, 1, current_plot)
        monthly_resample.plot()
        #plt.title(f'Monthly Trend of {column}',fontsize=5)
        plt.xlabel('Monthly',fontsize=5)
        plt.ylabel(column,fontsize=7)
        plt.yticks([])  # 隐藏y轴刻度值
        plt.tick_params(axis='x', which='both', labelbottom=False)  # Hide x-axis labels
        current_plot += 1
    
    if weekly:
        weekly_resample = df[column].resample('W').mean()
        plt.subplot(plot_count, 1, current_plot)
        weekly_resample.plot()
        #plt.title(f'Weekly Trend of {column}',fontsize=5)
        plt.xlabel('Weekly',fontsize=5)
        plt.ylabel('',fontsize=5)
        plt.yticks([])  # 隐藏y轴刻度值
        plt.tick_params(axis='x', which='both', labelbottom=False)  # Hide x-axis labels
        current_plot += 1
    
    if daily:
        daily_resample = df[column].resample('D').mean()
        plt.subplot(plot_count, 1, current_plot)
        daily_resample.plot()
        #plt.title(f'Daily Trend of {column}',fontsize=5)
        plt.xlabel('Daily',fontsize=5)
        plt.ylabel('',fontsize=5)
        plt.yticks([])  # 隐藏y轴刻度值
        plt.tick_params(axis='x', which='both', labelbottom=False)  # Hide x-axis labels
        current_plot += 1
    
    plt.tight_layout(pad=3.0)  # 自动调整子图间距
    plt.show()


def plot_decomposition_df(dataset,df, column): 
    if dataset == '2019_Xinjiang_Photovoltaic_Data.csv':  # 15 minutes
        df = df.resample('D').mean() #重采样——1D
        period = 96  # 24 hours * 4 periods 
    elif dataset ==  'Australia_Electricity_Load_and_Price_Forecasting_Data.csv':  # 30 minutes
        df = df.resample('W').mean() #重采样——1W
        period = 24   # 24 hours
    elif dataset == '2016 Electrical Engineering Mathematical Modeling Competition Load Forecasting Dataset.csv':  # 1 day
        period = 30  # Approximate days in a month
    else:
        raise ValueError('Invalid dataset name.')
    # 解耦
    decomposition = seasonal_decompose(df[column], model='additive',period=period)
    # 设置全局字体大小
    plt.rcParams.update({'font.size': 5})  # 将10替换为您希望的字体大小
    # 创建子图
    plt.figure(figsize=(6.93,3),dpi=300)
    # 观察结果
    plt.subplot(4, 1, 1)
    plt.plot(df[column], label='Original')
    
    plt.title(f'Data of {column} ')
    #plt.xlabel('Date')
    #plt.ylabel(column)
    plt.xticks([])  # 隐藏x轴刻度值
    plt.yticks([])  # 隐藏y轴刻度值
    plt.tick_params(axis='x', which='both', labelbottom=False)  # Hide x-axis labels
    plt.tick_params(axis='y', which='both', labelleft=False)  # Hide y-axis labels
    plt.legend()

    # 趋势成分
    plt.subplot(4, 1, 2)
    plt.plot(decomposition.trend, label='Trend', color='orange')
    #plt.title('Trend Component')
    #plt.xlabel('Date')
    #plt.ylabel('Trend')
    plt.tick_params(axis='x', which='both', labelbottom=False)  # Hide x-axis labels
    plt.tick_params(axis='y', which='both', labelleft=False)  # Hide y-axis labels
    plt.xticks([])  # 隐藏x轴刻度值
    plt.yticks([])  # 隐藏y轴刻度值
    plt.legend()

    # 季节性成分
    plt.subplot(4, 1, 3)
    plt.plot(decomposition.seasonal, label='Seasonal', color='green')
    #plt.title('Seasonal Component')
    #plt.xlabel('Date')
    #plt.ylabel('Seasonal')
    plt.xticks([])  # 隐藏x轴刻度值
    plt.yticks([])  # 隐藏y轴刻度值
    plt.tick_params(axis='x', which='both', labelbottom=False)  # Hide x-axis labels
    plt.tick_params(axis='y', which='both', labelleft=False)  # Hide y-axis labels
    plt.legend()

    # 残差成分
    plt.subplot(4, 1, 4)
    plt.plot(decomposition.resid, label='Residual', color='red')
    #plt.title('Residual Component')
    #plt.xlabel('Date')
    #plt.ylabel('Residual')
    plt.tick_params(axis='x', which='both', labelbottom=True)  # Show x-axis labels
    plt.tick_params(axis='y', which='both', labelleft=False)  # Hide y-axis labels
    plt.xticks([])  # 隐藏x轴刻度值
    plt.yticks([])  # 隐藏y轴刻度值
    plt.legend()

    plt.tight_layout()  # Adjust subplots to fit into figure area.
    plt.show()

def plot_corr_df(df, dpi=300):
    import seaborn as sns
    """
    Plots a heatmap of the correlation matrix for the given DataFrame.

    Parameters:
    df (pd.DataFrame): DataFrame containing the data to analyze.
    dpi (int): Dots per inch for the saved figure.
    """
    # 计算相关系数矩阵
    corr = df.corr()
    # 设置图像大小，这里可以根据需要调整
    plt.figure(figsize=(12, 12), dpi=dpi)
    
    ax=sns.heatmap(corr, annot=True,cmap='coolwarm', fmt=".2f", annot_kws={'size':7},linewidths=0.3)
    
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=7)

    # 调整x,y轴标签的字体大小
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=10)
    # 保存图像，使用tightbbox自动调整大小
    #plt.savefig('corr_heatmap.png', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    folder_path = r'C:/Users/think/Desktop/论文/LMLFF-lyy/LMLFF-code/datasets/Australia_Electricity_Load_and_Price_Forecasting_Data'
    dataset = 'Australia_Electricity_Load_and_Price_Forecasting_Data.csv'
    file_path = os.path.join(folder_path, dataset)
    
    raw_data = pd.read_csv(file_path, parse_dates=['date'], encoding='utf-8')
    raw_data.set_index('date', inplace=False)

    import numpy as np

    cols_data = raw_data.columns[1:]
    df_data = raw_data[cols_data]
    # 检查数据，并打印出 NaN 和 inf 值的位置
    if df_data.isnull().values.any():
        print("NaN values found in the dataset at the following locations:")
        print(df_data[df_data.isnull().any(axis=1)])
    # 将数据框转换为数值数组，以便检查 inf 值
    df_values = df_data.values.astype(float)
    if np.isinf(df_values).any():
        print("Inf values found in the dataset at the following locations:")
        print(df_data[np.isinf(df_values).any(axis=1)])
    # 替换 inf 和 -inf 为 NaN
    df_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    # 填充 NaN 值为均值
    df_data.fillna(raw_data.mean(), inplace=True)
    # 检查处理数据，并打印出 NaN 和 inf 值的位置
    if df_data.isnull().values.any():
        print("After processing, NaN values still found in the dataset at the following locations:")
        print(df_data[df_data.isnull().any(axis=1)])
    df_values = df_data.values.astype(float)
    if np.isinf(df_values).any():
        print("After processing, Inf values still  found in the dataset at the following locations:")
        print(raw_data[np.isinf(df_values ).any(axis=1)])


    task = 'corr_plot' # ['raw_plot', 'resample_plot',decomposition_plot]

    # 设置全局字体大小
    #plt.rcParams.update({'font.size': 5})  # 将10替换为您希望的字体大小


    if task == 'raw_plot':
        if dataset == '2016_Electrical_Engineering_Mathematical_Modeling_Competition_Load_Forecasting_Dataset.csv':
            plot_raw_df(raw_data, x='date', y_columns=['Max_temperature(℃)','Min_temperature(℃)','Average_temperature(℃)'])
            plot_raw_df(raw_data, x='date', y_columns=['Relative_humidity(average)'])
            plot_raw_df(raw_data, x='date', y_columns=['Rainfall(mm)'])
            plot_raw_df(raw_data, x='date', y_columns=['load'])
        elif dataset == 'Australia_Electricity_Load_and_Price_Forecasting_Data.csv':
            plot_raw_df(raw_data,x='date', y_columns=['dry bulb temperature(℃)','dew point temperature(℃)','wet bulb temperature(℃)'])
            plot_raw_df(raw_data, x='date', y_columns=['humidity'])
            plot_raw_df(raw_data, x='date', y_columns=['electrcity price'])
            plot_raw_df(raw_data, x='date', y_columns=['load'])
        elif dataset == '2019_Xinjiang_Photovoltaic_Data.csv':
            plot_raw_df(raw_data, x='date', y_columns=['components temperature(℃)','temperature(℃)'])
            plot_raw_df(raw_data, x='date', y_columns=['humidity(%)'])
            plot_raw_df(raw_data, x='date', y_columns=['air pressure(hPa)'])
            plot_raw_df(raw_data, x='date', y_columns=['total radiation(W/m2)','direct radiation(W/m2)','diffuse radiation(W/m2)'])
            plot_raw_df(raw_data, x='date', y_columns=['load(mw)'])
        else:
            print('No such dataset!')
    
    elif task == 'resample_plot':
        if dataset == '2016_Electrical_Engineering_Mathematical_Modeling_Competition_Load_Forecasting_Dataset.csv' or dataset == 'Australia_Electricity_Load_and_Price_Forecasting_Data.csv':
            for column in raw_data.columns:
                plot_resample_df(raw_data, column, daily=False)
        elif dataset == '2019_Xinjiang_Photovoltaic_Data.csv':
            for column in raw_data.columns:
                plot_resample_df(raw_data, column,yearly=False)
        else:
            print('No such dataset!')

    elif task == 'decomposition_plot':
        for column in raw_data.columns:
            plot_decomposition_df(dataset,raw_data, column)
    elif task == 'corr_plot':
        plot_corr_df(raw_data)
    else:
        print('No such task!')