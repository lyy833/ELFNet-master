# ELFNet

ELFNet is a deep learning framework for time series forecasting, designed to enhance prediction performance through contrastive learning and disentangled representations. It includes multiple model variants suitable for diverse forecasting tasks and scenarios.

## Features

- **Modular Design**: ELFNet adopts a modular architecture for easy extension and customization, supporting various time series forecasting models.
- **Contrastive Learning**: Enhances model representation capability and improves prediction accuracy via contrastive learning methods.
- **Disentangled Representations**: Employs disentangled representation learning to separate trend and seasonal components in time series.
- **Multi-Model Support**: Includes multiple model variants such as ELFNet_no_contrastive and ELFNet_no_disentanglement to meet different requirements.

## File Structure

```
ELFNet/
├── ELFNet.py                # ELFNet model definition
├── dilated_conv.py          # Dilated convolution module
├── data_process/            # Data processing module
├── datasets/                # Dataset files
├── exp_forecasting.py       # Forecasting experiment module
├── layers/                  # Network layer definitions
├── models/                  # Other model definitions
├── utils/                   # Utility functions
├── main.py                  # Main entry point
```

## Installation

1. Clone the repository:

   ```bash
   git clone https://gitee.com/lyy123666/ELFNet-master.git
   cd ELFNet-master
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Data Preparation: Place dataset files into the `datasets/` directory.
2. Configuration: Modify configuration parameters in `main.py` as needed.
3. Run Training:

   ```bash
   python main.py
   ```

## Datasets

ELFNet supports multiple time series datasets, including:

- Xinjiang Photovoltaic Dataset (`XJ_Photovoltaic.csv`)
- Australia Load & Price Forecasting Dataset (`Australia_Load&Price.csv`)
- 2016 Mathematical Modeling Competition Load Forecasting Dataset (`Mathematical_Modeling_Competition.csv`)
- Panama National Power Load Dataset (`Panama_CND.csv`)

## Model Variants

ELFNet provides several model variants, including:

- `ELFNet_no_contrastive`: Version without contrastive learning.
- `ELFNet_no_disentanglement`: Version without disentangled representations.
- `ELFNet_depthwise`: Version using depthwise separable convolutions.

## Tools

ELFNet offers various utility functions, including data augmentation, Dynamic Time Warping (DTW), metric calculation, and time feature extraction.

## Contribution

Contributions and suggestions are welcome. Please submit a Pull Request or open an Issue on Gitee.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.