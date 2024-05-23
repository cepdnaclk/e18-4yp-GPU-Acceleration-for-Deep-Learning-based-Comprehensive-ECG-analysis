import torch
import pandas as pd
from torch.utils.data import Dataset
import ast
import wfdb
import numpy as np
from scipy import signal as scipysignal
import os
import datetime
import socket
import utils.datasets as utils_datasets
from tqdm import tqdm
import random

DEFAULT = "default"
SHAPE_2D = "shape_2d"


# decide to run the full dataset or no based on the server or local machine
hostname = socket.gethostname()


def signal_lowpass_filter(signal, sampling_rate, cutoff_freq=30, order=4):
    nyquist_freq = 0.5 * sampling_rate
    normalized_cutoff = cutoff_freq / nyquist_freq
    b, a = scipysignal.butter(order, normalized_cutoff, btype="low", analog=False)
    filtered_signal = scipysignal.filtfilt(b, a, signal)
    return filtered_signal


class ECGDataset(Dataset):
    labels = ["MI", "STTC", "HYP", "NORM", "CD"]

    def __init__(self, path="./datasets/PTB_XL/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/", sampling_rate=500, input_shape=DEFAULT, num_of_leads=8):
        self.path = path
        self.sampling_rate = sampling_rate
        self.no_of_input_channels = input_shape
        self.num_of_leads = num_of_leads

        path_in_ram = f"/dev/shm/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/"
        if hostname == "ampere":
            print("Running in ampere server. Checking if data is available in ram")
            self.path = path_in_ram
            if os.path.exists(path_in_ram):
                print("Files found in ram. Continuing")
            else:
                print("Files not found in ram. Downloading now")
                utils_datasets.download_and_extract_ptb_xl_dataset_to_ram()
        else:
            print("Not running in ampere. Loading data from disk")
            self.path = path

        # Load and convert annotation data
        self.Y = pd.read_csv(self.path + "ptbxl_database.csv", index_col="ecg_id")
        self.Y.scp_codes = self.Y.scp_codes.apply(lambda x: ast.literal_eval(x))

        # normalize self.X
        # self.X = (self.X - np.min(self.X)) / (np.max(self.X) - np.min(self.X))

        # Load scp_statements.csv for diagnostic aggregation
        self.agg_df = pd.read_csv(path + "scp_statements.csv", index_col=0)
        self.agg_df = self.agg_df[self.agg_df.diagnostic == 1]

        # Apply diagnostic superclass and add the 'diagnostic_superclass' column

        self.Y["diagnostic_superclass"] = self.Y.scp_codes.apply(self.aggregate_diagnostic)
        self.Y = self.Y[self.Y["diagnostic_superclass"].apply(lambda x: len(x) == 1)]

        # Load raw signal data
        print("loading raw data")
        self.X = self.load_raw_data()

        print("normalizing each lead")
        self.X = self.normalize_each_lead(self.X)

        print("init done")

        # standardize self.X
        # self.X = (self.X - np.mean(self.X)) / np.std(self.X)

    def __len__(self):
        #  return 1000
        return len(self.Y)

    def __getitem__(self, idx):
        if self.no_of_input_channels == DEFAULT:
            x = torch.Tensor(self.X[idx].flatten())
        elif self.no_of_input_channels == SHAPE_2D:
            x = torch.Tensor(self.X[idx]).transpose(0, 1)  # Transpose the tensor to (8, 5000)
        y = self.Y["diagnostic_superclass"].iloc[idx][0]
        y = torch.tensor([y == i for i in self.labels], dtype=torch.float32)
        return x, y

    def normalize_each_lead(self, ecg_signals):
        for i in range(ecg_signals.shape[0]):  # Iterate through each sample
            for j in range(ecg_signals.shape[2]):  # Iterate through each lead
                signal = ecg_signals[i, :, j]
                min_val = np.min(signal)
                max_val = np.max(signal)
                ecg_signals[i, :, j] = (signal - min_val) / (max_val - min_val)  # Normalize lead
        return ecg_signals

    def load_raw_data(self):
        if self.num_of_leads == 8:
            channels_to_extract = [0, 1, 6, 7, 8, 9, 10, 11]
        elif self.num_of_leads == 12:
            channels_to_extract = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

        if self.sampling_rate == 100:
            data = [wfdb.rdsamp(self.path + f, channels=channels_to_extract) for f in tqdm(self.Y.filename_lr)]
        else:
            data = [wfdb.rdsamp(self.path + f, channels=channels_to_extract) for f in tqdm(self.Y.filename_hr)]

        data = np.array([signal for signal, _ in data])  # here _ is metadata : leftout

        return data
        # comment above code to get the low passed signals

        # Apply low-pass filter to smooth the waveform
        filtered_data = []
        for signal in data:
            filtered_channels = []
            for channel in range(signal.shape[1]):
                filtered_signal = signal_lowpass_filter(signal[:, channel], self.sampling_rate)
                filtered_channels.append(filtered_signal)
            filtered_data.append(np.column_stack(filtered_channels))

        filtered_data = np.array(filtered_data)
        return filtered_data

    def aggregate_diagnostic(self, y_dic):
        tmp = []
        for key in y_dic.keys():
            if key in self.agg_df.index:
                tmp.append(self.agg_df.loc[key].diagnostic_class)
        return list(set(tmp))


# Function to print with timestamp and time difference
def print_with_timestamp(message):
    global last_print_time
    current_time = datetime.now()
    if "last_print_time" not in globals():
        time_diff = None
    else:
        time_diff = (current_time - last_print_time).total_seconds()
    timestamp = current_time.strftime("[%Y-%m-%d %H:%M:%S]")
    if time_diff is not None:
        print(f"{timestamp} (Time Taken: {time_diff:.2f} seconds) {message}")
    else:
        print(f"{timestamp} {message}")
    last_print_time = current_time


# Example usage: if the path is different and the sampling rate different
# default set to current directory and sampleRate 100
# path = './'
# sampling_rate = 100
# ecg_dataset = ECGDataset(path, sampling_rate)

# ecg_dataset = ECGDataset()

# test to get the return shape of x and y

if __name__ == "__main__":

    # Initialize the dataset
    dataset_path = "./ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/"
    dataset = ECGDataset(path=dataset_path)

    # Retrieve a sample
    sample_index = 0  # For example, get the first sample
    x, y = dataset[sample_index]

    # Print sizes of x and y
    print(f"Size of x: {x.size()}")
    print(f"Size of y: {y.size()}")

    # Optionally, print x and y to see their values (might be large)
    # print(x)
    # print(y)
