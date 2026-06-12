# ELFNet: Electric Load Forecasting Based on Time Series Representation Learning

[![License: MPL 2.0](https://img.shields.io/badge/License-MPL_2.0-blue.svg)](https://opensource.org/licenses/MPL-2.0)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-orange.svg)](https://pytorch.org/)

## Overview

ELFNet (Evolving-Learning Fourier Network) is a deep time series representation learning model for electric load forecasting. This work takes time series representation learning as the main line and conducts progressive research from two perspectives: the design of representation extraction methods and the optimization of the learning paradigm.

At the representation extraction level, a **power time series representation extraction method based on variable grouping and component disentanglement (supervised ELFNet)** is proposed. It achieves multi-variable differentiated correlation modeling through a Hybrid Channel Setting-based Feature Extraction Module (HCSFEM), and achieves refined characterization and dynamic interaction of multi-scale temporal components through a Seasonal-Trend Disentanglement Module (DSTDM).

At the learning paradigm level, a **causal-aware and domain knowledge-guided self-supervised representation learning paradigm for power time series (self-supervised ELFNet)** is proposed. Through a Positive-Negative Contrastive learning framework with causal awareness and domain knowledge (PNCo), it constructs strongly discriminative samples with clear physical semantics, driving the model to learn feature representations robust to distribution shifts and sensitive to causal structures and domain-specific characteristics.

The model supports both self-supervised pretraining (via contrastive learning) and supervised training paradigms, and can be fine-tuned on target datasets through a two-stage training strategy.

This repository contains the official implementation of ELFNet and all baseline models used in the corresponding master's thesis.

## Key Features

- **HCSFEM (Hybrid Channel Setting-based Feature Extraction Module):** Pioneering adaptive hybrid channel settings in the power domain. It employs a Variable Adaptive Grouping algorithm (VAG) based on comprehensive similarity metrics and dynamic threshold hierarchical clustering to group variables, combined with group convolution-based DTCN, achieving refined multi-variable differentiated correlation modeling.
- **DSTDM (Seasonal-Trend Disentanglement Module):** Comprises a Trend Representation Disentangler (TRD, multi-scale causal convolutions) and a Seasonal Representation Disentangler (SRD, multi-band learnable Fourier layers). The **Multi-Band Learnable Fourier Layers**, unlike existing frequency-domain methods that model a single frequency band, partition the spectrum into multiple sub-bands with independently learnable parameters and incorporate an attention mechanism for adaptive fusion, enabling adaptive seasonal pattern learning. Additionally, a novel seasonal-trend bidirectional modulation mechanism (CGU) is proposed to achieve dynamic interaction between the two components for the first time in time series forecasting.
- **PNCo (Positive-Negative Contrastive learning with causal awareness and domain knowledge):** Integrates three strategies—causal-aware augmentation (variable importance enhancement and component-specific perturbation), domain-specific augmentation (load peak-valley enhancement), and global diversity augmentation—to perform contrastive learning on trend and seasonal components in both time and frequency domains, driving the model to learn robust and physically meaningful representations.
- **Two-Stage Training Strategy:** Self-supervised pretraining (contrastive) followed by supervised fine-tuning, with differentiated parameter freezing and weight transfer mechanisms supporting cross-dataset transfer learning.
- **Comprehensive Baselines:** Includes 10 baseline methods: TimesNet, Informer, DLinear, PatchTST, TS2Vec, CoST, TimeMAE, SegRNN, ADDSTCN, and PatchTST (supervised).

## Architecture Overview

```
Input (B, T, C)
    │
    ├─ HCSFEM (Hybrid Channel Setting-based Feature Extraction Module)
    │   ├─ Input Projection (per-variable Linear)
    │   └─ MixedChannelConvEncoder (grouped dilated convolutions)
    │
    ├─ DSTDM (Seasonal-Trend Disentanglement Module)
    │   ├─ Trend Representation Disentangler (multi-kernel causal conv)
    │   │   └── Multi-scale trend features
    │   ├─ Seasonal Representation Disentangler (multi-band Fourier layers)
    │   │   └── Frequency-domain seasonal features
    │   └─ Coupled Gating Unit (bidirectional modulation)
    │
    ├─ FeatureReducer (transposed conv)
    ├─ Projection & Pooling
    └─ Output (B, pred_len, 1)
```

## Project Structure

```
├── main.py                         # Entry point for training/testing
├── exp_forecasting.py              # Experiment orchestration (train/val/test)
├── requirements.txt                # Python dependencies
├── datasets/                       # Dataset storage (CSV format)
│   ├── datasets_readme.md          # Dataset descriptions
│   ├── XJ_Photovoltaic.csv
│   ├── Australia_Load&Price.csv
│   ├── Mathematical_Modeling_Competition.csv
│   └── Panama_CND.csv
├── data_process/
│   ├── data_provider.py            # DataLoader factory
│   └── custom_dataset.py           # Custom Dataset class
├── models/
│   ├── ELFNet.py                   # ELFNet (self-supervised & supervised)
│   ├── ELFNet_ablation.py          # Ablation variants of ELFNet
│   ├── DLinear.py / TimesNet.py / Informer.py / ...  # Baseline models
│   └── ...
├── layers/
│   ├── dilated_conv.py             # Dilated convolution blocks
│   ├── Embed.py                    # Time feature embeddings
│   ├── SelfAttention_Family.py     # Attention mechanisms
│   └── ...
├── utils/
│   ├── augmentation.py             # Domain augmentation framework
│   ├── tools.py                    # Training utilities (early stopping, etc.)
│   ├── metrics.py                  # Evaluation metrics
│   ├── variableGrouping.py         # Variable adaptive grouping algorithm
│   └── timefeatures.py             # Time feature extraction
└── experiments_analysis/           # (Ignored — personal analysis scripts)
```

## Installation

### Prerequisites

- Python 3.9 or later
- PyTorch 1.10+

### Setup

```bash
# Clone the repository
git clone https://github.com/<your-username>/ELFNet.git
cd ELFNet

# Install dependencies
pip install -r requirements.txt
```

## Datasets

This project supports four datasets (see `datasets/datasets_readme.md` for details):

| Dataset | Resolution | Span | Target Variable |
|---|---|---|---|
| XJ Photovoltaic | 15 min | 1 year | Load (MW) |
| Australia Load & Price | 30 min | 4 years | Load (MW) |
| Mathematical Modeling Competition | 1 day | 3 years | Load (MW) |
| Panama CND | 1 hour | 5+ years | National Demand (MW) |

Place CSV files in the `datasets/` directory. The dataset path is specified via the `--data_path` argument.

## Usage

### Quick Start (Single-Stage Supervised)

```bash
python main.py \
    --model_used DLinear \
    --data_path datasets/Australia_Load&Price.csv \
    --seq_len 96 --pred_len 48 \
    --finetune_target_idx 5
```

### ELFNet: Self-Supervised Pretraining + Fine-Tuning

**Stage 1 — Self-supervised pretraining:**

```bash
python main.py \
    --model_used ELFNet \
    --pretrain_data_path datasets/Panama_CND.csv \
    --seq_len 168 --pred_len 24 \
    --pretrain_target_idx 1 \
    --train_epochs1 20 \
    --training_mode single
```

**Stage 2 — Fine-tuning (requires a pretrained checkpoint):**

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

### ELFNet: Supervised Training (Single-Stage)

```bash
python main.py \
    --model_used ELFNet_supervised \
    --data_path datasets/Mathematical_Modeling_Competition.csv \
    --seq_len 90 --pred_len 30 \
    --finetune_target_idx 5 \
    --epochs 30
```

### ELFNet: Supervised Pretraining (Two-Stage Supervised)

```bash
# Stage 1 — Supervised pretraining
python main.py \
    --model_used ELFNet_supervised_pretrain \
    --pretrain_data_path datasets/Panama_CND.csv \
    --seq_len 168 --pred_len 24 \
    --pretrain_target_idx 1 \
    --train_epochs1 20

# Stage 2 — Fine-tuning
python main.py \
    --model_used ELFNet_supervised_pretrain \
    --data_path datasets/XJ_Photovoltaic.csv \
    --seq_len 192 --pred_len 96 \
    --finetune_target_idx 5 \
    --finetune_pretrained_model True \
    --pretrained_model_path ./test_results/.../pretrained_ELFNet_family/ELFNet_supervised_pretrain.pth \
    --train_epochs2 30
```

### Ablation Models

```bash
# Without trend-seasonal disentanglement
python main.py --model_used ELFNet_wo_TS --data_path <path> ...

# Without multi-band SRD (single-band Fourier layer)
python main.py --model_used ELFNet_single_band_SRD --data_path <path> ...

# Without Coupled Gating Unit
python main.py --model_used ELFNet_wo_CGU --data_path <path> ...

# With common seasonal-trend decomposition (moving average)
python main.py --model_used ELFNet_common_TS --data_path <path> ...
```

### Baseline Models

Supported models: `TimesNet`, `Informer`, `DLinear`, `PatchTST_SS`, `PatchTST_SU`, `SegRNN`, `ADDSTCN`, `TS2Vec`, `CoST`, `TimeMAE`.

```bash
python main.py --model_used TimesNet --data_path <path> --seq_len <L> --pred_len <P>
```

### Key Arguments

| Argument | Description | Default |
|---|---|---|
| `--model_used` | Model name | `PatchTST_SU` |
| `--seq_len` | Input sequence length | 90 |
| `--pred_len` | Prediction horizon | 30 |
| `--data_path` | Dataset path | `datasets/Mathematical_Modeling_Competition.csv` |
| `--pretrain_data_path` | Pretraining dataset (one2many) | `datasets/Mathematical_Modeling_Competition.csv` |
| `--training_mode` | `single` or `one2many` | `single` |
| `--train_epochs1` | Pretraining epochs (stage 1) | 20 |
| `--train_epochs2` | Fine-tuning epochs (stage 2) | 30 |
| `--finetune_pretrained_model` | Enable fine-tuning from pretrained checkpoint | False |
| `--batch_size` | Batch size | 64 |
| `--lr` | Learning rate | 0.00001 |
| `--hidden_dims` | Hidden dimension in feature extractor | 64 |
| `--repr_dims` | Representation dimension | 320 |
| `--use_gpu` | Use GPU | False |
| `--pretrain_target_idx` | Target variable index (pretrain) | 5 |
| `--finetune_target_idx` | Target variable index (finetune) | 5 |

For a full list of arguments, see `main.py`.

## Evaluation Metrics

The test script reports the following metrics:

- **MAE / RMSE:** Mean absolute and root mean squared error
- **NMAE / NRMSE:** Normalized variants
- **MAPE:** Mean absolute percentage error
- **R²:** Coefficient of determination
- **MASE:** Mean absolute scaled error
- **Peak Metrics:** Peak absolute error, peak time shift
- **Correlation:** Pearson correlation coefficient
- **Efficiency:** Inference time (ms) and model parameter count

## Citation

If you find this work useful for your research, please cite the corresponding thesis:

```bibtex
@mastersthesis{ELFNet2025,
  author  = {<Author Name>},
  title   = {Research on Short-Term Load Forecasting Based on Learnable
             Frequency-Domain Decomposition and Contrastive Representation Learning},
  school  = {Tianjin University},
  year    = {2025}
}
```

## License

This project is licensed under the [Mozilla Public License 2.0](LICENSE).

## Acknowledgement

This repository references code from the following open-source projects:

- [Autoformer](https://github.com/thuml/Autoformer)
- [Informer](https://github.com/zhouhaoyi/Informer2020)
- [TimesNet](https://github.com/thuml/TimesNet)
- [DLinear](https://github.com/cure-lab/LTSF-Linear)
- [PatchTST](https://github.com/yuqinie98/PatchTST)
- [TS2Vec](https://github.com/zhihanyue/ts2vec)
- [CoST](https://github.com/patrick-troy/CoST)
- [ADDSTCN](https://github.com/hfawaz/dl-4-tsc)
