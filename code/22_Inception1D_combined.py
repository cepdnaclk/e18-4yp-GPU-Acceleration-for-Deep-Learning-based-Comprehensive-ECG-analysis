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

from datasets.PTB_XL.PTB_XL_ECG_Dataset import ECGDataset
from models.Inception1D import Inception1dCombined
from datasets.PTB_XL.PTB_XL_ECG_Dataset import SHAPE_2D

start_time = time.time()

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
model1 = torch.load("saved_models/22_Inception1D.py_qt_20240523_111357_colorful-sun-362", map_location="cuda:0")  # HR
model2 = torch.load("saved_models/22_Inception1D.py_pr_20240523_111347_fast-surf-361", map_location="cuda:0")  # QRS
model3 = torch.load("saved_models/22_Inception1D.py_qrs_20240523_111328_cerulean-smoke-360", map_location="cuda:0")  # PR
model4 = torch.load("saved_models/22_Inception1D.py_hr_20240523_111112_winter-tree-359", map_location="cuda:0")  # QT

# model1 = Inception1d()
# model2 = Inception1d()
# model3 = Inception1d()
# model4 = Inception1d()

# Freeze the parameters of the pre-trained models
for param in model1.parameters():
    param.requires_grad = False

for param in model2.parameters():
    param.requires_grad = False

for param in model3.parameters():
    param.requires_grad = False

for param in model4.parameters():
    param.requires_grad = False

# Remove the last layer (MLP) from each model
model1.layers[1].pop(8)
model2.layers[1].pop(8)
model3.layers[1].pop(8)
model4.layers[1].pop(8)

# Hyperparameters
batch_size = 32
learning_rate = 0.01
num_epochs = 50
train_fraction = 0.8

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="version3_classification",
    # track hyperparameters and run metadata
    config={
        "learning_rate": learning_rate,
        "architecture": os.path.basename(__file__),
        "dataset": "PTB-XL",
        "epochs": num_epochs,
        "parameter": "classification",
    },
    notes="pretrained 4 inception1d models with deepfake. last layer removed. combined in a new model. with new linear layer at end",
)

# Create the model
model = Inception1dCombined(model1, model2, model3, model4)
model = model.to(device)

# Create the dataset class
dataset = ECGDataset(input_shape=SHAPE_2D, num_of_leads=8)

# Split the dataset into training and validation sets
train_size = int(train_fraction * len(dataset))
test_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

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
        }
    )

    # # Save the trained model with date and time in the path
    # current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # model_path = f"saved_models/{current_time}"
    # mlflow.pytorch.save_model(model, model_path)

print("Finished Training")


# Record the end time
end_time = time.time()

# Calculate and print the runtime
runtime = end_time - start_time
print(f"Runtime: {runtime} seconds")
