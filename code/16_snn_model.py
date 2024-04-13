import snntorch as snn
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models import SpikingRNNRegressor
from datasets.deepfake_ecg.Deepfake_ECG_Dataset import Deepfake_ECG_Dataset

from datasets.deepfake_ecg.Deepfake_ECG_Dataset import Deepfake_ECG_Dataset
from datasets.deepfake_ecg.Deepfake_ECG_Dataset import HR_PARAMETER
from datasets.deepfake_ecg.Deepfake_ECG_Dataset import QRS_PARAMETER
from datasets.deepfake_ecg.Deepfake_ECG_Dataset import PR_PARAMETER
from datasets.deepfake_ecg.Deepfake_ECG_Dataset import QT_PARAMETER

# Hyperparameters
batch_size = 32
learning_rate = 0.01
num_epochs = 50
parameter = HR_PARAMETER  # or QRS_PARAMETER, PR_PARAMETER, QT_PARAMETER

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create the dataset
dataset = Deepfake_ECG_Dataset(parameter=parameter)

# Split the dataset into training and validation sets
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))])

# Create data loaders
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Create the model
input_size = 40000 #dataset.data.shape[-1]
hidden_size = 128
num_steps = dataset.data.shape[1]
model = SpikingRNNRegressor(input_size, hidden_size, num_steps).to(device)

# Optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.L1Loss()

# Training loop
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for inputs, labels in train_dataloader:
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
        for inputs, labels in val_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item()

    # Print training and validation loss
    train_loss /= len(train_dataloader)
    val_loss /= len(val_dataloader)
    print(f"Epoch: {epoch + 1}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

# Save the trained model
torch.save(model.state_dict(), "saved_model.pt")