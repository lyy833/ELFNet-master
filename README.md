# ELFNet

ELFNet 是一个用于时间序列预测的深度学习框架，专注于通过对比学习和解耦表示来提高预测性能。它包含多种模型变体，适用于不同的预测任务和场景。

## 特性

- **模块化设计**：ELFNet 采用模块化设计，便于扩展和定制，支持多种时间序列预测模型。
- **对比学习**：通过对比学习方法增强模型的表示能力，提升预测准确性。
- **解耦表示**：使用解耦表示学习，分离时间序列中的趋势和季节性成分。
- **多模型支持**：包含多种模型变体，如 ELFNet_no_contrastive、ELFNet_no_disentanglement 等，适用于不同需求。

## 文件结构

```
ELFNet/
├── ELFNet.py                # ELFNet 模型定义
├── dilated_conv.py          # 膨胀卷积模块
├── data_process/            # 数据处理模块
├── datasets/                # 数据集文件
├── exp_forecasting.py       # 预测实验模块
├── layers/                  # 网络层定义
├── models/                  # 其他模型定义
├── utils/                   # 工具函数
├── main.py                  # 主程序入口
├── test_results/            # 结果保存文件夹  
```
其中 `test_results` 文件夹说明如下：
1. **一级文件夹**：以实验 setting 命名，例如 `Australia_Load&Price_seq_96_pred_48_stride_1_cluster_None_None`,里面包含的数据集名称是预训练数据集名称。
2. **二级文件夹**包括：
 - `plot` ：**可示化结果文件夹**，其下包含预训练数据集的增强可视化结果以及第一阶段和第二阶段的训练损失可视化，第一阶段的损失由于只涉及当前预训练数据集，直接命名为 `stage1_loss.png`;第二阶段的训练损失需要根据微调数据集设置名称，比如 `Mathematical_Modeling_Competition_stage2_loss.png`
 - `pretrained_ELFNet_family`：保存 `ELFNet` 及其消融模型预训练模型权重 `.pth`文件，比如 `ELFNet.pth`、`ELFNet_no_disentanglement.pth`等
 - `trained_compare_model`：baseline 模型训练权重 `.pth`文件保存路径，比如 `ELFNet.pth`、`ELFNet_no_disentanglement.pth`等
 - `finetuned_ELFNet_family`：其下又包含若干个**三级文件夹**，分别为 `ELFNet`,`ELFNet_depthwise`, `ELFNet_no_disentanglement`, `ELFNet_no_contrastive`, `ELFNet_dilution`，每个三级文件夹下面的微调模型以微调数据集命名,同时保存该微调模型的测试可视化结果。


## 安装

1. 克隆仓库：

   ```bash
   git clone https://gitee.com/lyy123666/ELFNet-master.git
   cd ELFNet-master
   ```

2. 安装依赖：

   ```bash
   pip install -r requirements.txt
   ```

## 使用

1. 数据准备：将数据集文件放入 `datasets/` 目录。
2. 修改配置：根据需要修改 `main.py` 中的配置参数。
3. 运行训练：

   ```bash
   python main.py
   ```

## 数据集

ELFNet 支持多种时间序列数据集，包括：

- 新疆光伏数据集 (`XJ_Photovoltaic.csv`)
- 澳大利亚电力负荷与价格预测数据 (`Australia_Load&Price.csv`)
- 2016电工数学建模竞赛负荷预测数据集（`Mathematical_Modeling_Competition.csv`）
- 巴拿马国家电力负荷数据集（`Panama_CND.csv`）

## 模型变体

ELFNet 提供多种模型变体，包括：

- `ELFNet_no_contrastive`: 不使用对比学习的版本。
- `ELFNet_no_disentanglement`: 不使用解耦表示的版本。
- `ELFNet_depthwise`: 使用深度可分离卷积的版本。

## 工具

ELFNet 提供多种工具函数，包括数据增强、动态时间规整（DTW）、指标计算、时间特征提取等。

## 贡献

欢迎贡献代码和建议。请提交 Pull Request 或在 Gitee 上提出 Issue。

## 许可证

该项目遵循 MIT 许可证。详情请查看 [LICENSE](LICENSE) 文件。