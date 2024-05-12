import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import datetime
import wandb
import os
import numpy as np
import random
from sklearn.model_selection import train_test_split

from models.TransformerForECGFeatures import TransformerEncoderModel
from datasets.deepfake_ecg.Deepfake_ECG_Dataset import Deepfake_ECG_Dataset
from datasets.deepfake_ecg.Deepfake_ECG_Dataset import HR_PARAMETER
from datasets.deepfake_ecg.Deepfake_ECG_Dataset import QRS_PARAMETER
from datasets.deepfake_ecg.Deepfake_ECG_Dataset import PR_PARAMETER
from datasets.deepfake_ecg.Deepfake_ECG_Dataset import QT_PARAMETER
import utils.current_server as current_server
import datasets.deepfake_ecg.Deepfake_ECG_Dataset as deepfake_ecg_dataset


parameter = HR_PARAMETER

# Hyperparameters
batch_size = 1
learning_rate = 0.01
num_epochs = 50
train_fraction = 0.8
patch_size = 3
input_size = 10
num_layers = 4
num_heads = 5
dim_feedforward = 2048
output_size = 1

best_model = None
best_validation_loss = 1000000

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="version2",
    # track hyperparameters and run metadata
    config={
        "learning_rate": learning_rate,
        "architecture": os.path.basename(__file__),
        "dataset": "Deepfake_ECG_Dataset",
        "epochs": num_epochs,
    },
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Create the model
model = TransformerEncoderModel(input_size, num_layers, num_heads, dim_feedforward, output_size).to(device)

# Create the dataset class
# Load all_lead_intervals here and provide it as an argument
dataset = Deepfake_ECG_Dataset(parameter=parameter, output_type=deepfake_ecg_dataset.FEATURE_EXTRACTION_ADEEPA)

print(len(dataset))

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

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)



# Feature Extraction 
import numpy as np
from scipy.signal import find_peaks

 # With batch size
# def process_ecg_intervals(ecg_data):
#     print("ecg_data.shape : ",ecg_data.shape )
#     """
#     Process an 8-lead .asc file to extract PR, RT, and PT intervals for each signal in a batch.
    
#     Args:
#     ecg_data (numpy.ndarray): Input array of shape (5000, 8, batch_size), where 5000 rows of data, 8 columns for each lead, and batch_size is the number of signals in the batch.
    
#     Returns:
#     numpy.ndarray: Array containing PR, RT, and PT intervals for each lead and each signal in the batch.
#     """
#     def extract_p_peaks(ecg_lead, amplitude_range=(50, 150), peaks=None):
#         if peaks is None:
#             peaks, _ = find_peaks(ecg_lead, distance=350)

#         p_peaks = []
#         for i in range(len(peaks) - 1):
#             r_peak = peaks[i]
#             r_next_peak = peaks[i+1]
            
#             window_start = r_peak - 200
#             window_end = r_peak - 50
            
#             window_peaks, _ = find_peaks(ecg_lead[window_start:window_end], height=amplitude_range)
            
#             p_peaks.extend([window_start + peak for peak in window_peaks])
        
#         return p_peaks

#     def extract_t_peaks(ecg_lead, amplitude_range=(150, 250), peaks=None):
#         if peaks is None:
#             peaks, _ = find_peaks(ecg_lead, distance=350)

#         t_peaks = []
#         for i in range(len(peaks) - 1):
#             r_peak = peaks[i]
#             r_next_peak = peaks[i+1]
            
#             window_start = r_peak + 50
#             window_end = r_next_peak - 200
            
#             window_peaks, _ = find_peaks(ecg_lead[window_start:window_end], height=amplitude_range)
            
#             t_peaks.extend([window_start + peak for peak in window_peaks])
        
#         return t_peaks

#     def calculate_intervals(ecg_lead, p_peaks, r_peaks, t_peaks, sampling_rate):
#         intervals = []
#         for i, r_peak in enumerate(r_peaks):
#             if i == 0 or i == len(r_peaks) - 1:
#                 continue

#             p_peak_idx = np.argmax(np.array(p_peaks) < r_peak)
#             p_peak = p_peaks[p_peak_idx]

#             t_peak_idx = np.argmax(np.array(t_peaks) > r_peak)
#             t_peak = t_peaks[t_peak_idx]

#             pr_interval = (r_peak - p_peak) / sampling_rate
#             rt_interval = (t_peak - r_peak) / sampling_rate
#             pt_interval = (t_peak - p_peak) / sampling_rate

#             intervals.append([pr_interval, rt_interval, pt_interval])

#         return np.array(intervals)

#     sampling_rate = 500
#     lead_intervals = []

#     for batch_idx in range(ecg_data.shape[2]):
#         ecg_data_selected = ecg_data[:, :, batch_idx]
#         batch_lead_intervals = []

#         for i, ecg_lead in enumerate(ecg_data_selected.T):
#             peaks, _ = find_peaks(ecg_lead, distance=350)
#             p_peaks = extract_p_peaks(ecg_lead, amplitude_range=(50, 150), peaks=peaks)
#             t_peaks = extract_t_peaks(ecg_lead, amplitude_range=(150, 250), peaks=peaks)
#             intervals = calculate_intervals(ecg_lead, p_peaks, peaks, t_peaks, sampling_rate)
#             batch_lead_intervals.append(intervals)

#         lead_intervals.append(batch_lead_intervals)

#     all_lead_intervals = np.array(lead_intervals)
#     print("all_lead_intervals.shape : ",all_lead_intervals.shape )
#     return all_lead_intervals


# Without batch size
def process_ecg_intervals(ecg_data):
    """
    Process an 8-lead .asc file to extract PR, RT, and PT intervals.
    
    Args:
    file_path (str): Path to the .asc file.
    
    Returns:
    numpy.ndarray: Array containing PR, RT, and PT intervals for each lead.
    """
    def extract_p_peaks(ecg_lead, amplitude_range=(50, 150), peaks=None):
        if peaks is None:
            peaks, _ = find_peaks(ecg_lead, distance=350)

        p_peaks = []
        for i in range(len(peaks) - 1):
            r_peak = peaks[i]
            r_next_peak = peaks[i+1]
            
            window_start = r_peak - 200
            window_end = r_peak - 50
            
            window_peaks, _ = find_peaks(ecg_lead[window_start:window_end], height=amplitude_range)
            
            p_peaks.extend([window_start + peak for peak in window_peaks])
        
        return p_peaks

    def extract_t_peaks(ecg_lead, amplitude_range=(150, 250), peaks=None):
        if peaks is None:
            peaks, _ = find_peaks(ecg_lead, distance=350)

        t_peaks = []
        for i in range(len(peaks) - 1):
            r_peak = peaks[i]
            r_next_peak = peaks[i+1]
            
            window_start = r_peak + 50
            window_end = r_next_peak - 200
            
            window_peaks, _ = find_peaks(ecg_lead[window_start:window_end], height=amplitude_range)
            
            t_peaks.extend([window_start + peak for peak in window_peaks])
        
        return t_peaks

    def calculate_intervals(ecg_lead, p_peaks, r_peaks, t_peaks, sampling_rate):
        intervals = []
        for i, r_peak in enumerate(r_peaks):
            if i == 0 or i == len(r_peaks) - 1:
                continue

            p_peak_idx = np.argmax(np.array(p_peaks) < r_peak)
            p_peak = p_peaks[p_peak_idx]

            t_peak_idx = np.argmax(np.array(t_peaks) > r_peak)
            t_peak = t_peaks[t_peak_idx]

            pr_interval = (r_peak - p_peak) / sampling_rate
            rt_interval = (t_peak - r_peak) / sampling_rate
            pt_interval = (t_peak - p_peak) / sampling_rate

            intervals.append([pr_interval, rt_interval, pt_interval])

        return np.array(intervals)

    sampling_rate = 500
    # ecg_data = np.loadtxt(file_path)
    selected_leads = [0, 1, 4, 5, 6, 7]
    ecg_data_selected = ecg_data[:, selected_leads]
    lead_intervals = []

    for i, ecg_lead in enumerate(ecg_data_selected.T):
        peaks, _ = find_peaks(ecg_lead, distance=350)
        p_peaks = extract_p_peaks(ecg_lead, amplitude_range=(50, 150), peaks=peaks)
        t_peaks = extract_t_peaks(ecg_lead, amplitude_range=(150, 250), peaks=peaks)
        intervals = calculate_intervals(ecg_lead, p_peaks, peaks, t_peaks, sampling_rate)
        lead_intervals.append(intervals)

    all_lead_intervals = np.array(lead_intervals)
    return all_lead_intervals

# # Example usage
# file_path = "./124.asc"
# ecg_data = np.loadtxt(file_path)
# all_lead_intervals = process_ecg_intervals(ecg_data)
# print(all_lead_intervals.shape)


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
        print("##############################################---------------------------############# inputs.shape : ", inputs.shape)
        
        # print("########################################################### inputs : ", inputs)
        inputs = inputs.squeeze()

        # file_path = "D:\SEM_07\FYP\e18-4yp-GPU-Acceleration-for-Deep-Learning-based-Comprehensive-ECG-analysis\code\others/124.asc"
        # inputs = np.loadtxt(file_path)
        print("########################################################### Before FUNCTION CALL inputs : ", inputs)
        print("########################################################### type(inputs) : ", type(inputs))
        print("##############################################---------------------------############# inputs.shape : ", inputs.shape)

        inputs = process_ecg_intervals(inputs.numpy())                           # FUNCTION CALL
        print("########################################################### After FUNCTION CALL inputs : ", inputs)
        print("##############################################@@@@@@@@@@@@@@@@############# inputs.shape : ", inputs.shape)
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
            inputs = process_ecg_intervals(inputs)                             # FUNCTION CALL
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    # Log metrics
    wandb.log({"train_loss": train_loss / (len(train_dataloader) * batch_size), "val_loss": val_loss / (len(val_dataloader) * batch_size)})

    print(f"Epoch: {epoch} train_loss: {train_loss / (len(train_dataloader)*batch_size)}")
    print(f"Epoch: {epoch} val_loss: {val_loss / (len(val_dataloader)*batch_size)}")

    if (val_loss / (len(val_dataloader) * batch_size)) < best_validation_loss:
        best_validation_loss = val_loss
        best_model = model

# Save the trained model with date and time in the path
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
model_path = f"saved_models/{os.path.basename(__file__)}_{current_time}_{wandb.run.name}"

torch.save(best_model, model_path)
print("Best Model Saved")
print("Finished Training")
wandb.finish()

# create
