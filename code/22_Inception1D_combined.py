import utils.others as others

print(f"Last updated by: ", others.get_latest_update_by())
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

from torch.optim.lr_scheduler import ExponentialLR
from sklearn.model_selection import train_test_split

from models.Inception1D import Inception1dCombined, Inception1d

from datasets.PTB_XL.PTB_XL_ECG_Dataset import ECGDataset, SHAPE_2D

from datasets.PTB_XL_Plus.PTB_XL_PLUS_ECG_Dataset import PTB_XL_PLUS_ECGDataset, SUB_DATASET_A, SUB_DATASET_B

start_time = time.time()

# Hyperparameters
batch_size = 31
learning_rate = 0.01
num_epochs = 1000
train_fraction = 0.8       # so test fraction is 0.2
val_fraction = 0.1         # val fraction is 0.1 out of the total dataset | 0.125 out of train fraction
select_sub_dataset = SUB_DATASET_A

patience = 50
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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load the pre-trained models
# TODO : add the correct paths to here >>>
# Load the pre-trained models with correct device mapping
# model1 = torch.load("saved_models/22_Inception1D.py_qt_20240523_111357_colorful-sun-362", map_location="cuda:0")  # HR
# model2 = torch.load("saved_models/22_Inception1D.py_pr_20240523_111347_fast-surf-361", map_location="cuda:0")  # QRS
# model3 = torch.load("saved_models/22_Inception1D.py_qrs_20240523_111328_cerulean-smoke-360", map_location="cuda:0")  # PR
# model4 = torch.load("saved_models/22_Inception1D.py_hr_20240523_111112_winter-tree-359", map_location="cuda:0")  # QT

model1 = Inception1d(num_classes=1, input_channels=8, use_residual=True, ps_head=0.5, lin_ftrs_head=[128], kernel_size=40).to(device)
model2 = Inception1d(num_classes=1, input_channels=8, use_residual=True, ps_head=0.5, lin_ftrs_head=[128], kernel_size=40).to(device)
model3 = Inception1d(num_classes=1, input_channels=8, use_residual=True, ps_head=0.5, lin_ftrs_head=[128], kernel_size=40).to(device)
model4 = Inception1d(num_classes=1, input_channels=8, use_residual=True, ps_head=0.5, lin_ftrs_head=[128], kernel_size=40).to(device)

# Freeze the parameters of the pre-trained models
# for param in model1.parameters():
#     param.requires_grad = False

# for param in model2.parameters():
#     param.requires_grad = False

# for param in model3.parameters():
#     param.requires_grad = False

# for param in model4.parameters():
#     param.requires_grad = False

# Remove the last layer (MLP) from each model
model1.layers[1].pop(8)
model2.layers[1].pop(8)
model3.layers[1].pop(8)
model4.layers[1].pop(8)

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="", # TODO : add a project name here
    # track hyperparameters and run metadata
    config={
        "learning_rate": learning_rate,
        "architecture": os.path.basename(__file__),
        "dataset": "PTB-XL-PLUS",
        "epochs": num_epochs,
        "parameter": "classification",
        "sub_dataset": select_sub_dataset
    },
    notes="""""",
)

# Create the model
model = Inception1dCombined(model1, model2, model3, model4)
model = model.to(device)

# Create the dataset class
# dataset = ECGDataset(input_shape=SHAPE_2D, num_of_leads=8)
dataset = PTB_XL_PLUS_ECGDataset(num_of_leads=8, sub_dataset=select_sub_dataset, is_classification=True)

# Split the dataset into training and validation sets
train_indices, test_indices = train_test_split(range(len(dataset)), test_size=1 - train_fraction, random_state=42, shuffle=True)
train_indices, val_indices = train_test_split(train_indices, test_size=(val_fraction/train_fraction), random_state=42, shuffle=True)

train_dataset = torch.utils.data.Subset(dataset, train_indices)
val_dataset = torch.utils.data.Subset(dataset, val_indices)
test_dataset = torch.utils.data.Subset(dataset, test_indices)

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
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# Optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()  # Use CrossEntropyLoss for multi-class classification
scheduler = ExponentialLR(optimizer, gamma=0.99)  # Set the gamma value for exponential decay

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
    scheduler.step()
    
    # Validation loop
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
        print(f"********Early stopping at epoch {epoch}********")
        break

#end of trainging Start of Testing
print("Using best model for Testing...")
best_model.eval()
total_correct = 0
total_samples = 0
all_outputs = []
all_labels = []

with torch.no_grad():
    for i, data in tqdm(
        enumerate(test_dataloader, 0),
        total=len(test_dataloader),
        desc=f"Testing",
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
    test_accuracy = total_correct / total_samples

    # Compute AUC-ROC
    all_outputs = np.concatenate(all_outputs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    test_auc_roc = roc_auc_score(all_labels, all_outputs)

    # Log metrics
    print(f"test_accuracy: {test_accuracy}, test_auc_roc: {test_auc_roc}, total_correct: {total_correct}, total_samples: {total_samples}")
#  Log metrics
wandb.log(
    {
        "test_accuracy": test_accuracy,
        "test_AUC": test_auc_roc,
    }
)


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