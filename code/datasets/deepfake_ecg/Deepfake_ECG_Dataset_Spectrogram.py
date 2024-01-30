import torch
import pandas
from scipy.signal import spectrogram
import numpy as np

import socket

# decide to run the full dataset or no based on the server or local machine
hostname = socket.gethostname()

server_hostnames = ["ampere", "turing", "aiken"]
if hostname in server_hostnames:
    # run full dataset on servers
    IS_FULL_DATASET = True
    print("Running full dataset")
else:
    # limit to 1000 on local computers
    IS_FULL_DATASET = False
    print("Running limited dataset")

HR_PARAMETER = "hr"
QRS_PARAMETER = "qrs"
PR_PARAMETER = "pr"
QT_PARAMETER = "qt"


class Deepfake_ECG_Dataset_Spectrogram(torch.utils.data.Dataset):
    """
    Deepfake ECG dataset filtered to only include normal ECGs
    Contains 121977 ECGs
    ECG signals are returned as a 1D tensor (40k numbers)
    Parameters are returned as a 1D tensor
    """

    def __init__(self, parameter=None):
        super(Deepfake_ECG_Dataset_Spectrogram, self).__init__()

        if parameter not in [HR_PARAMETER, QRS_PARAMETER, PR_PARAMETER, QT_PARAMETER]:
            raise ValueError("Invalid parameter")

        # load the ground truth labels
        self.ground_truths = pandas.read_csv(
            "datasets/deepfake_ecg/filtered_all_normals_121977_ground_truth.csv"
        )

        if parameter == HR_PARAMETER:
            parameter = torch.tensor(
                self.ground_truths["avgrrinterval"].values, dtype=torch.float32
            )
            # calculate HR
            self.parameter = 60 * 1000 / parameter
        elif parameter == QRS_PARAMETER:
            self.parameter = torch.tensor(
                self.ground_truths["qrs"].values, dtype=torch.float32
            )
        elif parameter == PR_PARAMETER:
            self.parameter = torch.tensor(
                self.ground_truths["pr"].values, dtype=torch.float32
            )
        elif parameter == QT_PARAMETER:
            self.parameter = torch.tensor(
                self.ground_truths["qt"].values, dtype=torch.float32
            )

        # Dictionary to store loaded ASC files
        self.loaded_asc_files = {}

    def __getitem__(self, index):
        filename = self.ground_truths["patid"].values[index]

        # Check if the ASC file is already loaded
        if filename in self.loaded_asc_files:
            ecg_signals = self.loaded_asc_files[filename]
        else:
            # Load the ASC file
            ecg_signals = pandas.read_csv(
                f"datasets/deepfake_ecg/from_006_chck_2500_150k_filtered_all_normals_121977/{filename}.asc",
                header=None,
                sep=" ",
            )

            # files have 8 columns, each column has one lead
            # extract each lead and flatten it
            # then append the flattened lead to allData so that allData has lead1, lead2, lead3, etc. one after the other
            allData = []
            for column in ecg_signals.columns:
                ecg_array = ecg_signals[column].values
                flattened_array = ecg_array.flatten()
                allData.extend(flattened_array)
            ecg_signals = torch.tensor(allData, dtype=torch.float32)
            ecg_signals = ecg_signals.reshape(-1)

            _, _, Sxx = spectrogram(ecg_signals, 500)
            # Sxx = 10 * np.log10(Sxx)
            ecg_signals = Sxx.reshape(-1)

            # Store the loaded ASC file in the dictionary
            self.loaded_asc_files[filename] = ecg_signals

        parameter = self.parameter[index].reshape(-1)

        return ecg_signals, parameter

    def __len__(self):
        if IS_FULL_DATASET:
            # run full dataset on servers
            return self.ground_truths.shape[0]

        # limit to 1000 on local computers
        return 1000


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    ## Create an instance of the dataset with the 'hr' parameter
    ecg_dataset = Deepfake_ECG_Dataset_Spectrogram(parameter="hr")

    ## Create a data loader for the dataset
    batch_size = 1
    data_loader = DataLoader(
        ecg_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )

    ## Iterate over a few batches and print the shapes of X and y
    for batch_idx, (X, y) in enumerate(data_loader):
        print(f"Batch {batch_idx + 1}:")
        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")
        print("\n")

        ## Perform any additional testing or analysis here

        ## Break the loop after a few batches for demonstration purposes
        if batch_idx == 2:
            break
