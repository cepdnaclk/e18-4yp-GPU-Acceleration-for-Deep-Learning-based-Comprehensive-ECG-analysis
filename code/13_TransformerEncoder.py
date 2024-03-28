import utils.others as others
print(f"Last updated by: ",others.get_latest_update_by())
# Code 2: Running the Transformer Model (Modified)

import torch
import torch.nn as nn
from tqdm import tqdm
import datetime
import wandb
import os
import numpy as np
import random
from sklearn.model_selection import train_test_split

from models.TransformerEncoderModel import TransformerEncoderModel
from datasets.deepfake_ecg.Deepfake_ECG_Dataset import Deepfake_ECG_Dataset
from datasets.deepfake_ecg.Deepfake_ECG_Dataset import HR_PARAMETER
from datasets.deepfake_ecg.Deepfake_ECG_Dataset import QRS_PARAMETER
from datasets.deepfake_ecg.Deepfake_ECG_Dataset import PR_PARAMETER
from datasets.deepfake_ecg.Deepfake_ECG_Dataset import QT_PARAMETER
import utils.current_server as current_server


# Hyperparameters
batch_size = 1
learning_rate = 0.001
num_epochs = 1000
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
    
best_model = None
best_validation_loss = 1000000

# import torch
# import torch.nn as nn
# from datasets.deepfake_ecg.Deepfake_ECG_Dataset import Deepfake_ECG_Dataset, HR_PARAMETER
# from tqdm import tqdm

# Hyperparameters
batch_size = 32
learning_rate = 0.001
num_epochs = 50
train_fraction = 0.8
patch_size = 500
input_size = 512
num_layers = 4
num_heads = 8
dim_feedforward = 2048

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="version2",
    # track hyperparameters and run metadata
    config={
        "learning_rate": learning_rate,
        "architecture": os.path.basename(__file__),
        "dataset": "Deepfake_ECG_Dataset",
        "epochs": num_epochs,
        "parameter": parameter,
    },
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Create the model
model = TransformerEncoderModel(input_size, patch_size, num_layers, num_heads, dim_feedforward, output_size=1).to(device)

# Create the dataset class
dataset = Deepfake_ECG_Dataset(parameter=parameter)

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


# Optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.L1Loss()


# Training loop
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for i, data in tqdm(
        enumerate(train_dataloader, 0),
        total=len(train_dataloader),
        desc=f"Training Epoch {epoch + 1}/{num_epochs}",
    ):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # Validation loop
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for i, data in tqdm(
            enumerate(val_dataloader, 0),
            total=len(val_dataloader),
            desc=f"Validating Epoch {epoch + 1}/{num_epochs}",
        ):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            # if i == 0:
            #     for x in range(len(outputs)):
            #         print(f"Predicted: {outputs[x]} Real: {labels[x]}")
            loss = criterion(outputs, labels)

            val_loss += loss.item()

    #  Log metrics
    wandb.log({"train_loss": train_loss / (len(train_dataloader) * batch_size), "val_loss": val_loss / (len(val_dataloader) * batch_size)})

    print(f"Epoch: {epoch} train_loss: {train_loss / (len(train_dataloader)*batch_size)}")
    print(f"Epoch: {epoch} val_loss: {val_loss / (len(val_dataloader)*batch_size)}")
    
    if (val_loss / (len(val_dataloader) * batch_size)) < best_validation_loss:
        best_validation_loss = val_loss
        best_model = model

# Save the trained model with date and time in the path
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
model_path = f"saved_models/{os.path.basename(__file__)}_{parameter}_{current_time}"

torch.save(best_model, model_path)
print("Best Model Saved")
print("Finished Training")
wandb.finish()

# create a backup of mlruns in babbage server
# "Turing is not stable, data could be lost" - Akila E17
import os

os.system("cp -r mlruns ~/4yp/")
