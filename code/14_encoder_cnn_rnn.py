import torch
import torch.nn as nn
from tqdm import tqdm
import datetime
import wandb
import os

from models.Encoder_CNN_RNN import TransformerEncoderModel
from models.Encoder_CNN_RNN import CNN_RNN_Model
from datasets.deepfake_ecg.Deepfake_ECG_Dataset import Deepfake_ECG_Dataset
from datasets.deepfake_ecg.Deepfake_ECG_Dataset import HR_PARAMETER

# Hyperparameters
batch_size = 32
max_lr = 0.001
num_epochs = 1000
train_fraction = 0.8
parameter = HR_PARAMETER
patch_size = 500
input_size = 512
num_layers = 8
num_heads = 16
dim_feedforward = 2048
dropout = 0.1

# Exponential learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR

# Start a new wandb run to track this script
wandb.init(
    project="ecg-transformer",
    config={
        "max_lr": max_lr,
        "architecture": os.path.basename(__file__),
        "dataset": "Deepfake_ECG_Dataset",
        "epochs": num_epochs,
        "parameter": parameter,
        "patch_size": patch_size,
        "input_size": input_size,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "dim_feedforward": dim_feedforward,
        "dropout": dropout,
    },
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Create the models
transformer_model = TransformerEncoderModel(input_size, patch_size, num_layers, num_heads, dim_feedforward, output_size=1, dropout=dropout).to(device)
cnn_rnn_model = CNN_RNN_Model(input_size=patch_size, output_size=1).to(device)

# Create the dataset class
dataset = Deepfake_ECG_Dataset(parameter=HR_PARAMETER)

# Split the dataset into training and validation sets
train_size = int(train_fraction * len(dataset))
test_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# Create data loaders for training and validation
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# Optimizers and loss function
transformer_optimizer = torch.optim.Adam(transformer_model.parameters(), lr=max_lr)
cnn_rnn_optimizer = torch.optim.Adam(cnn_rnn_model.parameters(), lr=max_lr)
criterion = nn.L1Loss()

# Learning rate schedulers
transformer_scheduler = lr_scheduler(transformer_optimizer, num_epochs, eta_min=1e-6)
cnn_rnn_scheduler = lr_scheduler(cnn_rnn_optimizer, num_epochs, eta_min=1e-6)

# Training loop
for epoch in range(num_epochs):
    transformer_model.train()
    cnn_rnn_model.train()
    train_loss = 0.0
    for i, data in tqdm(enumerate(train_dataloader, 0), total=len(train_dataloader), desc=f"Training Epoch {epoch + 1}/{num_epochs}"):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        transformer_optimizer.zero_grad()
        cnn_rnn_optimizer.zero_grad()

        transformer_outputs = transformer_model(inputs)
        cnn_rnn_outputs = cnn_rnn_model(inputs.view(-1, patch_size, 1))

        ensemble_outputs = (transformer_outputs + cnn_rnn_outputs) / 2
        loss = criterion(ensemble_outputs, labels)

        loss.backward()
        transformer_optimizer.step()
        cnn_rnn_optimizer.step()

        train_loss += loss.item()

    print(f"Epoch: {epoch} train_loss: {train_loss / len(train_dataloader)}")

    # Validation loop
    transformer_model.eval()
    cnn_rnn_model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for i, data in tqdm(enumerate(val_dataloader, 0), total=len(val_dataloader), desc=f"Validating Epoch {epoch + 1}/{num_epochs}"):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            transformer_outputs = transformer_model(inputs)
            cnn_rnn_outputs = cnn_rnn_model(inputs.view(-1, patch_size, 1))

            ensemble_outputs = (transformer_outputs + cnn_rnn_outputs) / 2
            loss = criterion(ensemble_outputs, labels)

            val_loss += loss.item()

        print(f"Epoch: {epoch} val_loss: {val_loss / len(val_dataloader)}")

    # Log metrics
    wandb.log(
        {
            "train_loss": train_loss / len(train_dataloader),
            "val_loss": val_loss / len(val_dataloader),
        }
    )

    # Update learning rate schedulers
    transformer_scheduler.step()
    cnn_rnn_scheduler.step()

# Save the trained models with date and time in the path
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
transformer_model_path = f"saved_models/transformer_{current_time}"
cnn_rnn_model_path = f"saved_models/cnn_rnn_{current_time}"
torch.save(transformer_model, transformer_model_path)
torch.save(cnn_rnn_model, cnn_rnn_model_path)

print("Finished Training")
wandb.finish()