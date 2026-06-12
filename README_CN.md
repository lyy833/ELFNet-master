# ELFNet: 基于时间序列表示学习的电力负荷预测模型

[![License: MPL 2.0](https://img.shields.io/badge/License-MPL_2.0-blue.svg)](https://opensource.org/licenses/MPL-2.0)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-orange.svg)](https://pytorch.org/)

## 概述

ELFNet (Evolving-Learning Fourier Network) 是一个面向电力负荷预测的深度时序预测学习模型。本文以时间序列表示学习为主线，从表示提取方法设计与学习范式优化两个层面开展递进式研究。

在表示提取方法层面，提出了**变量分组与成分解耦的电力时序表示提取方法（有监督版ELFNet）**，通过基于混合通道设置的特征提取模块（HCSFEM）实现多变量差异化关联建模，通过深度季节-趋势解耦模块（DSTDM）实现多尺度时序成分的精细刻画与动态交互。

在学习范式层面，提出了**因果感知与领域知识引导的电力时序自监督表示学习范式（自监督版ELFNet）**，通过正负增强对比学习框架（PNCo）构造具有明确物理语义的强判别性样本，驱使模型学习对分布变化鲁棒、对因果结构与领域特性敏感的特征表示。

模型支持自监督预训练（基于对比学习）和有监督训练两种范式，并可通过两阶段训练策略在目标数据集上进行迁移学习与微调。

本仓库包含对应硕士论文中 ELFNet 的官方实现及所有基线模型代码。

## 核心创新点

- **HCSFEM（混合通道设置特征提取模块）：** 首次将自适应式混合通道设置引入电力领域，通过基于综合相似度度量与动态阈值层次聚类的变量自适应分组算法（VAG）对变量进行分组，结合基于分组卷积的DTCN，实现多变量差异化关联的精细化建模。
- **DSTDM（深度季节-趋势解耦模块）：** 包含趋势表示提取器（TRD，多尺度因果卷积）与季节性表示提取器（SRD，多频带可学习傅里叶层），其中的**多频带可学习傅里叶层：** 区别于现有频域时序分析方法的单频带建模，将频谱划分为多个配置独立可学习参数的子带，并引入注意力机制自适应融合各子带信息，实现自适应季节性模式学习。此外，创新性地提出季节-趋势双向调制机制（CGU），首次在时序预测中实现两类成分的动态交互建模。
- **PNCo（因果感知与领域知识引导的正负增强对比学习框架）：** 包含因果感知增强（变量关键性增强与时序成分特异性增强）、领域场景增强（负荷峰谷增强）及全局样本多样性增强三大策略，在时域与频域分别对趋势与季节成分开展对比学习，驱使模型学习鲁棒且具有物理语义的时序表示。
- **两阶段训练策略：** 自监督预训练（对比学习）→ 有监督微调，引入差异化参数冻结与权重迁移机制，支持跨数据集迁移学习。
- **全面的基线模型：** 包含10种对比方法：TimesNet、Informer、DLinear、PatchTST、TS2Vec、CoST、TimeMAE、SegRNN、ADDSTCN、PatchTST（有监督版）。

## 网络架构

```
输入 (B, T, C)
    │
    ├─ HCSFEM（混合通道设置特征提取模块）
    │   ├─ 输入投影层（每个变量独立 Linear 映射）
    │   └─ MixedChannelConvEncoder（分组膨胀卷积）
    │
    ├─ DSTDM（深度季节-趋势解耦模块）
    │   ├─ 趋势表示解耦器 TRD（多核因果卷积）
    │   │   └── 多尺度趋势特征
    │   ├─ 季节性表示解耦器 SRD（多频带傅里叶层）
    │   │   └── 频域季节性特征
    │   └─ 耦合门控单元 CGU（双向调制）
    │
    ├─ FeatureReducer（转置卷积）
    ├─ 全连接投影层 & 自适应池化
    └─ 输出 (B, pred_len, 1)
```

## 项目结构

```
├── main.py                         # 训练/测试入口
├── exp_forecasting.py              # 实验流程编排（训练/验证/测试）
├── requirements.txt                # Python 依赖
├── datasets/                       # 数据集存放（CSV 格式）
│   ├── datasets_readme.md          # 数据集说明
│   ├── XJ_Photovoltaic.csv         # 新疆光伏数据集
│   ├── Australia_Load&Price.csv    # 澳大利亚电力负荷与价格数据
│   ├── Mathematical_Modeling_Competition.csv  # 电工数学建模竞赛数据
│   └── Panama_CND.csv              # 巴拿马国家电力负荷数据
├── data_process/
│   ├── data_provider.py            # DataLoader 工厂
│   └── custom_dataset.py           # 自定义 Dataset 类
├── models/
│   ├── ELFNet.py                   # ELFNet（自监督版 & 有监督版）
│   ├── ELFNet_ablation.py          # ELFNet 消融变体
│   ├── DLinear.py / TimesNet.py / Informer.py / ...  # 基线模型
│   └── ...
├── layers/
│   ├── dilated_conv.py             # 膨胀卷积块
│   ├── Embed.py                    # 时间特征嵌入
│   ├── SelfAttention_Family.py     # 注意力机制
│   └── ...
├── utils/
│   ├── augmentation.py             # 领域数据增强框架
│   ├── tools.py                    # 训练工具（早停、学习率调整等）
│   ├── metrics.py                  # 评估指标
│   ├── variableGrouping.py         # 变量自适应分组算法
│   └── timefeatures.py             # 时间特征提取
└── experiments_analysis/           # （忽略——个人实验分析脚本）
```

## 环境配置

### 依赖要求

- Python 3.9+
- PyTorch 1.10+

### 安装步骤

```bash
# 克隆仓库
git clone https://github.com/<your-username>/ELFNet.git
cd ELFNet

# 安装依赖
pip install -r requirements.txt
```

## 数据集

本项目支持四个数据集（详见 `datasets/datasets_readme.md`）：

| 数据集 | 时间粒度 | 时间跨度 | 预测目标 |
|---|---|---|---|
| 新疆光伏 | 15分钟 | 1年 | 发电功率 (MW) |
| 澳大利亚电力负荷与电价 | 30分钟 | 4年 | 电力负荷 (MW) |
| 电工数学建模竞赛 | 1天 | 3年 | 日需求负荷 (MW) |
| 巴拿马国家电力负荷 | 1小时 | 5年以上 | 国家需求负荷 (MW) |

将 CSV 文件放入 `datasets/` 目录，通过 `--data_path` 参数指定路径。

## 使用说明

### 快速开始（单阶段有监督训练）

```bash
python main.py \
    --model_used DLinear \
    --data_path datasets/Australia_Load&Price.csv \
    --seq_len 96 --pred_len 48 \
    --finetune_target_idx 5
```

### ELFNet：自监督预训练 + 微调

**阶段1 — 自监督预训练：**

```bash
python main.py \
    --model_used ELFNet \
    --pretrain_data_path datasets/Panama_CND.csv \
    --seq_len 168 --pred_len 24 \
    --pretrain_target_idx 1 \
    --train_epochs1 20 \
    --training_mode single
```

**阶段2 — 微调（需要预训练检查点）：**

```bash
python main.py \
    --model_used ELFNet \
    --data_path datasets/XJ_Photovoltaic.csv \
    --seq_len 192 --pred_len 96 \
    --finetune_target_idx 5 \
    --finetune_pretrained_model True \
    --pretrained_model_path ./test_results/.../pretrained_ELFNet_family/ELFNet.pth \
    --train_epochs2 30
```

### ELFNet：有监督训练（单阶段）

```bash
python main.py \
    --model_used ELFNet_supervised \
    --data_path datasets/Mathematical_Modeling_Competition.csv \
    --seq_len 90 --pred_len 30 \
    --finetune_target_idx 5 \
    --epochs 30
```

### ELFNet：有监督预训练（两阶段有监督）

```bash
# 阶段1 — 有监督预训练
python main.py \
    --model_used ELFNet_supervised_pretrain \
    --pretrain_data_path datasets/Panama_CND.csv \
    --seq_len 168 --pred_len 24 \
    --pretrain_target_idx 1 \
    --train_epochs1 20

# 阶段2 — 微调
python main.py \
    --model_used ELFNet_supervised_pretrain \
    --data_path datasets/XJ_Photovoltaic.csv \
    --seq_len 192 --pred_len 96 \
    --finetune_target_idx 5 \
    --finetune_pretrained_model True \
    --pretrained_model_path ./test_results/.../pretrained_ELFNet_family/ELFNet_supervised_pretrain.pth \
    --train_epochs2 30
```

### 消融模型

```bash
# 移除季节-趋势解耦
python main.py --model_used ELFNet_wo_TS --data_path <path> ...

# 单频带傅里叶层（替换多频带SRD）
python main.py --model_used ELFNet_single_band_SRD --data_path <path> ...

# 移除耦合门控单元
python main.py --model_used ELFNet_wo_CGU --data_path <path> ...

# 传统滑动平均分解
python main.py --model_used ELFNet_common_TS --data_path <path> ...
```

### 基线模型

支持的模型：`TimesNet`、`Informer`、`DLinear`、`PatchTST_SS`、`PatchTST_SU`、`SegRNN`、`ADDSTCN`、`TS2Vec`、`CoST`、`TimeMAE`。

```bash
python main.py --model_used TimesNet --data_path <path> --seq_len <L> --pred_len <P>
```

### 关键参数说明

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--model_used` | 模型名称 | `PatchTST_SU` |
| `--seq_len` | 输入序列长度 | 90 |
| `--pred_len` | 预测步长 | 30 |
| `--data_path` | 数据集路径 | `datasets/Mathematical_Modeling_Competition.csv` |
| `--pretrain_data_path` | 预训练数据集路径 | `datasets/Mathematical_Modeling_Competition.csv` |
| `--training_mode` | 训练模式（single / one2many） | `single` |
| `--train_epochs1` | 预训练轮数（阶段1） | 20 |
| `--train_epochs2` | 微调轮数（阶段2） | 30 |
| `--finetune_pretrained_model` | 是否从预训练检查点微调 | False |
| `--batch_size` | 批大小 | 64 |
| `--lr` | 学习率 | 0.00001 |
| `--hidden_dims` | 特征提取器隐藏维度 | 64 |
| `--repr_dims` | 表示维度 | 320 |
| `--use_gpu` | 是否使用 GPU | False |
| `--pretrain_target_idx` | 预训练目标变量索引 | 5 |
| `--finetune_target_idx` | 微调目标变量索引 | 5 |

完整参数列表请参见 `main.py`。

## 评估指标

测试脚本输出以下指标：

- **MAE / RMSE：** 平均绝对误差 / 均方根误差
- **NMAE / NRMSE：** 归一化 MAE / RMSE
- **MAPE：** 平均绝对百分比误差
- **R²：** 决定系数
- **MASE：** 平均绝对缩放误差
- **峰值指标：** 峰值绝对误差、峰值时间偏移
- **相关系数：** 皮尔逊相关系数
- **效率指标：** 推理时间（毫秒）、模型参数量

## 引用

如果您的研究使用了本项目，请引用对应的硕士论文：

```bibtex
@mastersthesis{ELFNet2025,
  author  = {<作者名>},
  title   = {基于可学习频域分解与对比表示学习的短期负荷预测研究},
  school  = {天津大学},
  year    = {2025}
}
```

## 开源协议

本项目采用 [Mozilla Public License 2.0](LICENSE) 开源协议。

## 致谢

本仓库参考了以下开源项目的代码实现：

- [Autoformer](https://github.com/thuml/Autoformer)
- [Informer](https://github.com/zhouhaoyi/Informer2020)
- [TimesNet](https://github.com/thuml/TimesNet)
- [DLinear](https://github.com/cure-lab/LTSF-Linear)
- [PatchTST](https://github.com/yuqinie98/PatchTST)
- [TS2Vec](https://github.com/zhihanyue/ts2vec)
- [CoST](https://github.com/patrick-troy/CoST)
- [ADDSTCN](https://github.com/hfawaz/dl-4-tsc)
