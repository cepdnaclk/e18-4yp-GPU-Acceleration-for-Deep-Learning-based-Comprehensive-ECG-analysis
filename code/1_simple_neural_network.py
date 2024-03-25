import torch
import torch.nn as nn
from tqdm import tqdm
import mlflow
import mlflow.pytorch
import time
from models.SimpleNeuralNetwork import SimpleNeuralNetwork
from datasets.deepfake_ecg.Deepfake_ECG_Dataset import Deepfake_ECG_Dataset
from datasets.deepfake_ecg.Deepfake_ECG_Dataset import HR_PARAMETER
from datasets.deepfake_ecg.Deepfake_ECG_Dataset import QRS_PARAMETER
from datasets.deepfake_ecg.Deepfake_ECG_Dataset import PR_PARAMETER
from datasets.deepfake_ecg.Deepfake_ECG_Dataset import QT_PARAMETER
from sklearn.model_selection import train_test_split
import utils.current_server as current_server


import numpy as np

# Record the start time
start_time = time.time()

# TODO: Replace mlflow with wandb

# Hyperparameters
batch_size = 32
learning_rate = 0.001
num_epochs = 50
train_fraction = 0.8
parameter = HR_PARAMETER

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Create the model
model = SimpleNeuralNetwork().to(device)

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

# MLflow setup
mlflow.set_experiment("SimpleNeuralNetwork")
with mlflow.start_run():
    # Log hyperparameters
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("num_epochs", num_epochs)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
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

            running_loss += loss.item()

        # Log metrics
        mlflow.log_metric("train_loss", running_loss / len(train_dataloader), step=epoch)
        print(f"Epoch: {epoch} train_loss: {running_loss / len(train_dataloader)}")

        # Validation loop
        model.eval()
        running_loss = 0.0
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

                running_loss += loss.item()

            # Log metrics
            mlflow.log_metric("val_loss", running_loss / len(val_dataloader), step=epoch)

        print(f"Epoch: {epoch} val_loss: {running_loss / len(val_dataloader)}")

    # # Save the trained model with date and time in the path
    # current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # model_path = f"saved_models/{current_time}"
    # mlflow.pytorch.save_model(model, model_path)

print("Finished Training")

# create a backup of mlruns in babbage server
# "Turing is not stable, data could be lost" - Akila E17
import os

os.system("cp -r mlruns ~/4yp/")

# Record the end time
end_time = time.time()

# Calculate and print the runtime
runtime = end_time - start_time
print(f"Runtime: {runtime} seconds")
