import utils.others as others

print(f"Last updated by: ", others.get_latest_update_by())

import torch
from tqdm import tqdm
import numpy as np
import random
from sklearn.model_selection import train_test_split

from datasets.deepfake_ecg.Deepfake_ECG_Dataset import Deepfake_ECG_Dataset
from datasets.deepfake_ecg.Deepfake_ECG_Dataset import HR_PARAMETER
from datasets.deepfake_ecg.Deepfake_ECG_Dataset import FEATURE_EXTRACTION_ADEEPA
import utils.current_server as current_server


# Hyperparameters
batch_size = 1
learning_rate = 0.01
num_epochs = 1
train_fraction = 0.8
parameter = HR_PARAMETER

# Set a fixed seed for reproducibility
SEED = 42

# Set the seed for CPU
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# Set the seed for CUDA (GPU)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


# Create the dataset class
dataset = Deepfake_ECG_Dataset(parameter=parameter, output_type=FEATURE_EXTRACTION_ADEEPA)

# Split the dataset into training and validation sets
train_indices, val_indices = train_test_split(range(len(dataset)), test_size=1 - train_fraction, random_state=42, shuffle=True)

train_dataset = torch.utils.data.Subset(dataset, train_indices)
val_dataset = torch.utils.data.Subset(dataset, val_indices)

# set num_workers
if current_server.is_running_in_server():
    print(f"Running in {current_server.get_current_hostname()} server, Settings num_workers to 4")
    num_workers = 4
else:
    print(f"Running in {current_server.get_current_hostname()} server, Settings num_workers to 0")
    num_workers = 0

# Create data loaders for training and validation
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# Training loop
try:
    for epoch in range(num_epochs):
        for i, data in tqdm(
            enumerate(train_dataloader, 0),
            total=len(train_dataloader),
            desc=f"Training Epoch {epoch + 1}/{num_epochs}",
        ):
            inputs, labels = data

        # Validation loop
        with torch.no_grad():
            for i, data in tqdm(
                enumerate(val_dataloader, 0),
                total=len(val_dataloader),
                desc=f"Validating Epoch {epoch + 1}/{num_epochs}",
            ):
                inputs, labels = data

        print("reading all data okay")
except Exception as e:
