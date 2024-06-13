import utils.others as others

print(f"Last updated by: ", others.get_latest_update_by())

logging_enabled = False
import torch
import torch.nn as nn
from tqdm import tqdm
import datetime
import wandb
import os
import time
import utils.current_server as current_server
import numpy as np
import random
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from torch.optim.lr_scheduler import CosineAnnealingLR

# Record the start time
start_time = time.time()

from models.Inception1D import Inception1d
from datasets.PTB_XL.PTB_XL_ECG_Dataset import ECGDataset, SHAPE_2D
from datasets.PTB_XL_Plus.PTB_XL_PLUS_ECG_Dataset import PTB_XL_PLUS_ECGDataset, SUB_DATASET_A, SUB_DATASET_B


# Hyperparameters
batch_size = 31
learning_rate = 0.01
num_epochs = 5
train_fraction = 0.8
select_subdataset = SUB_DATASET_A

patience = 30
early_stopping_counter = 0
best_val_auc_roc = 0

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

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="version3_classification",  #    project="final_runs_by_ridma",
    # track hyperparameters and run metadata
    config={"learning_rate": learning_rate, "architecture": os.path.basename(__file__), "dataset": "PTB-XL", "epochs": num_epochs, "parameter": "classification", "sub_dataset": select_subdataset},
    notes="classification no transfer learning 1 inception alone",
)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Create the model
model = Inception1d(num_classes=5, input_channels=12, use_residual=True, ps_head=0.5, lin_ftrs_head=[128], kernel_size=40).to(device)

# Create the dataset class
dataset = PTB_XL_PLUS_ECGDataset(num_of_leads=12, sub_dataset=select_subdataset, is_classification=True)


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
criterion = nn.CrossEntropyLoss()  # Use CrossEntropyLoss for multi-class classification
scheduler = CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-5)


for epoch in range(num_epochs):
    model.train()
    total_correct = 0
    total_samples = 0
    all_outputs = []
    all_labels = []

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

        if logging_enabled:
            try:
                print("label      |     output")
                for round in range(batch_size):
                    print(labels[round], "  |  ", outputs[round])
                print()
            except Exception as e:
                # Print the error message
                print("An error occurred at print label and output:", e)

        # Calculate accuracy
        predicted = torch.argmax(outputs, 1)
        labels_max = torch.argmax(labels, 1)
        total_correct += (predicted == labels_max).sum().item()
        total_samples += labels.size(0)

        # Store outputs and labels for AUC-ROC calculation
        all_outputs.extend(outputs.detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    # Compute accuracy
    train_accuracy = total_correct / total_samples

    # Compute AUC-ROC
    all_outputs = np.concatenate(all_outputs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    train_auc_roc = roc_auc_score(all_labels, all_outputs)

    # Log metrics
    print(f"Epoch: {epoch} train_accuracy: {train_accuracy}, train_auc_roc: {train_auc_roc}, total_correct: {total_correct}, total_samples: {total_samples}")
    # Validation loop
    # Update learning rate scheduler
    scheduler.step()

    model.eval()
    total_correct = 0
    total_samples = 0
    all_outputs = []
    all_labels = []

    with torch.no_grad():
        for i, data in tqdm(
            enumerate(val_dataloader, 0),
            total=len(val_dataloader),
            desc=f"Validating Epoch {epoch + 1}/{num_epochs}",
        ):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)

            # Calculate accuracy
            predicted = torch.argmax(outputs, 1)
            labels_max = torch.argmax(labels, 1)
            total_correct += (predicted == labels_max).sum().item()
            total_samples += labels.size(0)

            # Store outputs and labels for AUC-ROC calculation
            all_outputs.extend(outputs.detach().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        # Compute accuracy
        val_accuracy = total_correct / total_samples

        # Compute AUC-ROC
        all_outputs = np.concatenate(all_outputs, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        val_auc_roc = roc_auc_score(all_labels, all_outputs)

        # Log metrics
        print(f"Epoch: {epoch} val_accuracy: {val_accuracy}, val_auc_roc: {val_auc_roc}, total_correct: {total_correct}, total_samples: {total_samples}")
    #  Log metrics
    wandb.log(
        {
            "train_accuracy": train_accuracy,
            "train_AUC": train_auc_roc,
            "val_accuracy": val_accuracy,
            "val_AUC": val_auc_roc,
            "lr": scheduler.get_last_lr()[0],
        }
    )
    # Early stopping
    if val_auc_roc > best_val_auc_roc:
        best_val_auc_roc = val_auc_roc
        best_model = model
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1

    # Check if early stopping criteria is met
    if early_stopping_counter >= patience:
        print(f"Early stopping at epoch {epoch}")
        break


# Save the trained model with date and time in the path
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
model_path = f"saved_models/{os.path.basename(__file__)}_classification_{current_time}_{wandb.run.name}"

torch.save(best_model, model_path)
print("Finished Training")


# Record the end time
end_time = time.time()

# Calculate and print the runtime
runtime = end_time - start_time
print(f"Runtime: {runtime} seconds")
