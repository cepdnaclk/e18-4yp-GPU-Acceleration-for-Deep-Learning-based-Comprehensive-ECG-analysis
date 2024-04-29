import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import pandas
import torch

ecg_signals = pandas.read_csv(
    f"datasets/deepfake_ecg/from_006_chck_2500_150k_filtered_all_normals_121977/1.asc",
    header=None,
    sep=" ",
)

allData = []
for column in ecg_signals.columns:
    ecg_array = ecg_signals[column].values
    flattened_array = ecg_array.flatten()
    # apply min max scaling
    flattened_array = (flattened_array - min(flattened_array)) / (max(flattened_array) - min(flattened_array))
    allData.extend(flattened_array)
ecg_signals = torch.tensor(allData, dtype=torch.float32)
ecg_signals = ecg_signals.reshape(-1)

# Assume your ECG signal is stored in the array 'ecg_signal'
ecg_signal = np.array(ecg_signals)  # Replace with your actual signal data

# Original sampling rate
original_sampling_rate = 500

# Desired downsampling factor (e.g., 5 to reduce sampling rate to 100 Hz)
downsampling_factor = 5

# Calculate the new sampling rate
new_sampling_rate = original_sampling_rate / downsampling_factor

# Downsample the signal using scipy.signal.decimate
downsampled_signal = signal.decimate(ecg_signal, downsampling_factor)

# Print the original and downsampled signal lengths
print(f"Original signal length: {len(ecg_signal)}")
print(f"Downsampled signal length: {len(downsampled_signal)}")

figure, axis = plt.subplots(2)
axis[0].plot(ecg_signal)
axis[1].plot(downsampled_signal)
plt.show()
