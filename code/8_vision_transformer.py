import utils.others as others

print(f"Last updated by: ", others.get_latest_update_by())
import torch
import torch.nn as nn
from tqdm import tqdm
import datetime
import wandb
import os
import utils.current_server as current_server
import utils.gpu as gpu
from sklearn.model_selection import train_test_split


import torchvision.models.vision_transformer as vision_transformer
import datasets.deepfake_ecg.Deepfake_ECG_Dataset as deepfake_ecg_dataset

# Hyperparameters
batch_size = 32
learning_rate = 0.01
num_epochs = 50
train_fraction = 0.8
parameter = deepfake_ecg_dataset.HR_PARAMETER

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="version2",
    # track hyperparameters and run metadata
    config={
        "learning_rate": learning_rate,
        "architecture": os.path.basename(__file__),
        "dataset": "Deepfake vision transformer VISION_TRANSFORMER_IMAGE_OUTPUT_TYPE",
        "epochs": num_epochs,
        "parameter": parameter,
        "batch_size": batch_size,
    },
)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Create the model
# for more information on how to use the class. Read the source code
# https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py
model = vision_transformer.vit_b_16(num_classes=1).to(device)

# Create the dataset class
dataset = deepfake_ecg_dataset.Deepfake_ECG_Dataset(
    parameter=parameter,
    output_type=deepfake_ecg_dataset.VISION_TRANSFORMER_IMAGE_OUTPUT_TYPE,
)

# Split the dataset into training and validation sets
train_indices, val_indices = train_test_split(
    range(len(dataset)),
    test_size=1 - train_fraction,
    random_state=42,
    shuffle=True,
)

train_dataset = torch.utils.data.Subset(dataset, train_indices)
val_dataset = torch.utils.data.Subset(dataset, val_indices)


# set num_workers
if current_server.is_running_in_server():
    print(
        f"Running in {current_server.get_current_hostname()} server, Settings num_workers to 4"
    )
    num_workers = 4
else:
    print(
        f"Running in {current_server.get_current_hostname()} server, Settings num_workers to 0"
    )
    num_workers = 0

# Create data loaders for training and validation
train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
)
val_dataloader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
)

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

            loss = criterion(outputs, labels)

            val_loss += loss.item()

    #  Log metrics
    wandb.log(
        {
            "train_loss": train_loss
            / (len(train_dataloader) * batch_size),
            "val_loss": val_loss / (len(val_dataloader) * batch_size),
        }
    )

    print(
        f"Epoch: {epoch} train_loss: {train_loss /  (len(train_dataloader)*batch_size)}"
    )
    print(
        f"Epoch: {epoch} val_loss: {val_loss /  (len(val_dataloader)*batch_size)}"
    )

# Save the trained model with date and time in the path
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
model_path = f"saved_models/{current_time}"
torch.save(model, model_path)

print("Finished Training")
wandb.finish()

# create a backup of mlruns in babbage server
# "Turing is not stable, data could be lost" - Akila E17
import os

os.system("cp -r mlruns ~/4yp/")
