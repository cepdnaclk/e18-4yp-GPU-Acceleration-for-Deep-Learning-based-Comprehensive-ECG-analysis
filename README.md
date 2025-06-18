# GPU Acceleration for Deep Learning-based Comprehensive ECG Analysis

## Repository Structure

```
e18-4yp-GPU-Acceleration-for-Deep-Learning-based-Comprehensive-ECG-analysis/
├── README.md               
├── code/                   # Main code directory
│   ├── 22_Inception1D_classification.py          
│   ├── 22_Inception1D_regression.py
│   ├── 22_Inception1D_regression_to_classification.py
│   ├── 22_Inception1D_classification_to_regression.py
│   ├── 22_Inception1D_regression_to_regression.py
│   ├── ...                 
│   ├── datasets/           # Dataset loading and processing code
│   │   ├── PTB_XL/         
│   │   ├── PTB_XL_Plus/    
│   │   └── deepfake_ecg/   
│   ├── models/             # Model architectures
│   ├── utils/              # Utility functions
│   └── env_install.sh      # Environment setup script
├── docs/                   
└── others/                 
```

## Getting Started

### Prerequisites

- Python 3.x
- Conda 
- CUDA-enabled GPU (recommended)
- Weights & Biases account (for experiment tracking)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/cepdnaclk/e18-4yp-GPU-Acceleration-for-Deep-Learning-based-Comprehensive-ECG-analysis.git
   cd e18-4yp-GPU-Acceleration-for-Deep-Learning-based-Comprehensive-ECG-analysis
   ```

2. Create and activate the Conda environment:
   ```bash
   conda create -n ecg_analysis python=3.8
   conda activate ecg_analysis
   ```

3. Install dependencies:
   ```bash
   bash code/env_install.sh
   ```

4. Log in to Weights & Biases:
   ```bash
   wandb login
   ```
   (Follow the prompts to enter your API key)

## Training the Models

### Main Classification Model

To train the primary Inception1D classification model:
```bash
python3 code/22_Inception1D_classification.py
```

### Transfer Learning Model

To train the transfer learning model (from regression to classification):
```bash
python3 code/22_Inception1D_regression_to_classification.py
```

### Other Models

Various other models are available in the `code/` directory. You can train them similarly:
```bash
python3 code/[model_filename].py
```

## Datasets

The code automatically handles dataset downloading and preprocessing for:
- PTB-XL ECG Dataset
- PTB-XL+ ECG Dataset
- Deepfake ECG Dataset

### Dataset Notes
- Datasets will be downloaded automatically when first run
- For the datasets on server environments, data is loaded into RAM for faster access
- Local machines will use a limited subset of the data by default

## Configuration

Key parameters can be adjusted in the model files:
- `batch_size`: Training batch size (default: 31)
- `learning_rate`: Initial learning rate (default: 0.01)
- `num_epochs`: Maximum training epochs (default: 1000)
- `train_fraction`: Fraction of data for training (default: 0.8)
- `val_fraction`: Fraction of training data for validation (default: 0.1)
- `patience`: Early stopping patience (default: 50)

## Monitoring

Training is logged using:
- Weights & Biases (wandb) for experiment tracking
- Local log files 

## Saved Models

Trained models are automatically saved to the `saved_models/` directory with timestamps and wandb run names.

## Contact

For questions or issues, please contact:
- ridmajayasundara@eng.pdn.ac.lk
- e18098@eng.pdn.ac.lk
- e18100@eng.pdn.ac.lk
