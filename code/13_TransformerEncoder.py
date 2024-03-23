# Code 2: Running the Transformer Model (Modified)

import torch
import torch.nn as nn
from tqdm import tqdm
import datetime
import wandb
import os

from models.TransformerEncoderModel import TransformerEncoderModel
from datasets.deepfake_ecg.Deepfake_ECG_Dataset import Deepfake_ECG_Dataset
from datasets.deepfake_ecg.Deepfake_ECG_Dataset import HR_PARAMETER
from datasets.deepfake_ecg.Deepfake_ECG_Dataset import QRS_PARAMETER
from datasets.deepfake_ecg.Deepfake_ECG_Dataset import PR_PARAMETER
from datasets.deepfake_ecg.Deepfake_ECG_Dataset import QT_PARAMETER

# Hyperparameters
batch_size = 1
learning_rate = 0.001
num_epochs = 1000
train_fraction = 0.8
parameter = HR_PARAMETER

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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Create the model
model = TransformerEncoderModel(input_size, patch_size, num_layers, num_heads, dim_feedforward, output_size=1).to(device)

# Create the dataset class
dataset = Deepfake_ECG_Dataset(parameter=HR_PARAMETER)

# Split the dataset into training and validation sets
train_size = int(train_fraction * len(dataset))
test_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# Create data loaders for training and validation
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# Optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.L1Loss()

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, data in tqdm(enumerate(train_dataloader, 0), total=len(train_dataloader), desc=f"Training Epoch {epoch + 1}/{num_epochs}"):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch: {epoch} train_loss: {running_loss / len(train_dataloader)}")

    # Validation loop
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for i, data in tqdm(enumerate(val_dataloader, 0), total=len(val_dataloader), desc=f"Validating Epoch {epoch + 1}/{num_epochs}"):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

        print(f"Epoch: {epoch} val_loss: {running_loss / len(val_dataloader)}")

print("Finished Training")