import utils.others as others
print(f"Last updated by: ", others.get_latest_update_by())

import xgboost as xgb
from tqdm import tqdm
import wandb
import os
import numpy as np
import random
from sklearn.model_selection import train_test_split

from datasets.deepfake_ecg.Deepfake_ECG_Dataset import Deepfake_ECG_Dataset
from datasets.deepfake_ecg.Deepfake_ECG_Dataset import HR_PARAMETER
from datasets.deepfake_ecg.Deepfake_ECG_Dataset import QRS_PARAMETER
from datasets.deepfake_ecg.Deepfake_ECG_Dataset import PR_PARAMETER
from datasets.deepfake_ecg.Deepfake_ECG_Dataset import QT_PARAMETER
import utils.current_server as current_server

# Hyperparameters
learning_rate = 0.01
num_epochs = 50
train_fraction = 0.8
parameter = HR_PARAMETER

# Set a fixed seed for reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# start a new wandb run to track this script
wandb.init(
    project="version2",
    config={
        "learning_rate": learning_rate,
        "architecture": os.path.basename(__file__),
        "dataset": "Deepfake_ECG_Dataset",
        "epochs": num_epochs,
        "parameter": parameter,
    },
)

# Create the dataset class
dataset = Deepfake_ECG_Dataset(parameter=parameter)

# Split the dataset into training and validation sets
train_indices, val_indices = train_test_split(range(len(dataset)), test_size=1 - train_fraction, random_state=42, shuffle=True)

train_data = [dataset[i][0].numpy() for i in train_indices]
train_labels = [dataset[i][1].numpy() for i in train_indices]

val_data = [dataset[i][0].numpy() for i in val_indices]
val_labels = [dataset[i][1].numpy() for i in val_indices]

# Create XGBoost model
model = xgb.XGBRegressor(objective='reg:squarederror', random_state=SEED)

# Train the model
model.fit(
    train_data,
    train_labels,
    eval_set=[(val_data, val_labels)],
    eval_metric='rmse',
    verbose=True,
)

# Log the performance
val_predictions = model.predict(val_data)
val_rmse = np.sqrt(((val_predictions - val_labels) ** 2).mean())
wandb.log({"val_rmse": val_rmse})

print(f"Validation RMSE: {val_rmse}")
wandb.finish()