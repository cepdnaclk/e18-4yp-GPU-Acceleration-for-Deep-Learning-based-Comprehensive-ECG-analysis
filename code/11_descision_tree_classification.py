import utils.others as others
print(f"Last updated by: ",others.get_latest_update_by())
import torch
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import wandb
import os
import numpy as np
import random
import time
import utils.current_server as current_server
from datasets.PTB_XL.PTB_XL_ECG_Dataset import ECGDataset

# Record the start time
start_time = time.time()

# Hyperparameters
test_fraction = 0.2

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
    project="version2",
    config={
        "model": "DecisionTree",
        "dataset": "PTB-XL",
        "architecture": os.path.basename(__file__),
        "dataset": "PTB-XL",
        "parameter": "classification",
    },
)

# Create the dataset class
dataset = ECGDataset()

# Split the dataset into training and testing sets
train_data, test_data = train_test_split(dataset, test_size=test_fraction, random_state=42)

# Separate features and labels
train_features, train_labels = torch.stack([data[0] for data in train_data]), torch.stack([data[1] for data in train_data])
test_features, test_labels = torch.stack([data[0] for data in test_data]), torch.stack([data[1] for data in test_data])

# Flatten the features
train_features = train_features.view(train_features.size(0), -1).numpy()
test_features = test_features.view(test_features.size(0), -1).numpy()

# Create the Decision Tree model
model = DecisionTreeClassifier(random_state=42)

# Train the model
model.fit(train_features, train_labels)

# Predictions on the training set
train_predictions = model.predict(train_features)
train_accuracy = accuracy_score(train_labels, train_predictions)

# Predictions on the test set
test_predictions = model.predict(test_features)
test_accuracy = accuracy_score(test_labels, test_predictions)

# Log metrics
wandb.log(
    {
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
    }
)

print(f"Train Accuracy: {train_accuracy}")
print(f"Test Accuracy: {test_accuracy}")

# Record the end time
end_time = time.time()

# Calculate and print the runtime
runtime = end_time - start_time
print(f"Runtime: {runtime} seconds")
