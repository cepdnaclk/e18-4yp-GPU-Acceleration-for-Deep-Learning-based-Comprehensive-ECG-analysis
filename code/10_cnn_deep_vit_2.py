import utils.others as others
print(f"Last updated by: ",others.get_latest_update_by())
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import datetime
import wandb
import os

from datasets.deepfake_ecg.Deepfake_ECG_Dataset import Deepfake_ECG_Dataset
from models.CnnDeepViT_2 import CnnDeepViT

import datasets.deepfake_ecg.Deepfake_ECG_Dataset as deepfake_ecg_dataset
from datasets.deepfake_ecg.Deepfake_ECG_Dataset import HR_PARAMETER
from datasets.deepfake_ecg.Deepfake_ECG_Dataset import QRS_PARAMETER
from datasets.deepfake_ecg.Deepfake_ECG_Dataset import PR_PARAMETER
from datasets.deepfake_ecg.Deepfake_ECG_Dataset import QT_PARAMETER

# Hyperparameters
batch_size = 1
learning_rate = 0.02
num_epochs = 50
train_fraction = 0.8
validation_fraction = 0.2
parameter = HR_PARAMETER  # Define the parameter

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Initialize wandb run
wandb.init(
    project="initial-testing",
    config={
        "learning_rate": learning_rate,
        "architecture": os.path.basename(__file__),
        "dataset": "Deepfake_ECG_Dataset DEEP_VIT_GREY_256_IMAGE_OUTPUT_TYPE",
        "epochs": num_epochs,
        "parameter": parameter,
    },
)

# Load dataset
dataset = Deepfake_ECG_Dataset(
    parameter=parameter,
    output_type=deepfake_ecg_dataset.DEEP_VIT_GREY_256_IMAGE_OUTPUT_TYPE,
)

# Split dataset
train_size = int(train_fraction * len(dataset))
test_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(
    dataset, [train_size, test_size]
)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Initialize model
model = CnnDeepViT(image_size=256, num_classes=1).to(device)

# Define optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.L1Loss()

# Training loop
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs.float())
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # Validation loop
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs.float())
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    # Log metrics
    wandb.log({
        "train_loss": train_loss / len(train_loader),
        "val_loss": val_loss / len(val_loader),
    })

    print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss / len(train_loader)}, Val Loss: {val_loss / len(val_loader)}")

# Save the trained model
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
model_path = f"saved_models/{current_time}"
torch.save(model.state_dict(), model_path)

# Finish wandb run
wandb.finish()
