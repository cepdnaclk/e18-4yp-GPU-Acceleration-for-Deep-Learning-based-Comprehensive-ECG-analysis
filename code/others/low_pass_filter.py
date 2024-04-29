import matplotlib.pyplot as plt
from scipy import signal as scipysignal
import pandas
import torch


def signal_lowpass_filter(signal, sampling_rate, cutoff_freq=30, order=4):
    nyquist_freq = 0.5 * sampling_rate
    normalized_cutoff = cutoff_freq / nyquist_freq
    b, a = scipysignal.butter(order, normalized_cutoff, btype="low", analog=False)
    filtered_signal = scipysignal.filtfilt(b, a, signal)
    return filtered_signal


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


def generate_low_passed_signal(ecg_signals, cutoff_freq=40):
    ecg_signals = signal_lowpass_filter(ecg_signals, 500, cutoff_freq=cutoff_freq, order=4)
    ecg_signals = ecg_signals.copy()
    ecg_signals = torch.tensor(ecg_signals, dtype=torch.float32)

    # Transposing the ECG signals
    ecg_signals = ecg_signals
    ecg_signals = ecg_signals.t()
    return ecg_signals


# frquencies = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
frquencies = [50,60]

# Initialise the subplot function using number of rows and columns
figure, axis = plt.subplots(len(frquencies))

index = 0
for x in frquencies:
    ecg_signals_new = generate_low_passed_signal(ecg_signals, cutoff_freq=x)
    # show in subplot
    axis[index].plot(ecg_signals_new)
    index += 1

# plot ecg_signal using matplotlib
plt.show()
