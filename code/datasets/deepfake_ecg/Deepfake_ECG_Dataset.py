import torch
import pandas


class Deepfake_ECG_Dataset(torch.utils.data.Dataset):
    """
    Deepfake ECG dataset filtered to only include normal ECGs
    Contains 121977 ECGs
    ECG signals are returned as a 1D tensor (40k numbers)
    Parameters are returned as a 1D tensor
    """

    def __init__(self):
        super(Deepfake_ECG_Dataset, self).__init__()

        # load the ground truth labels
        self.ground_truths = pandas.read_csv(
            "datasets/deepfake_ecg/filtered_all_normals_121977_ground_truth.csv"
        )

        # TODO: take the column name as input and use it to select the column
        # hardcode for now to only get the HR
        RR = torch.tensor(
            self.ground_truths["avgrrinterval"].values, dtype=torch.float32
        )
        # calculate HR
        self.heart_rate = 60 * 1000 / RR

    def __getitem__(self, index):
        filename = self.ground_truths["patid"].values[index]
        ecg_signals = pandas.read_csv(
            f"datasets/deepfake_ecg/from_006_chck_2500_150k_filtered_all_normals_121977/{filename}.asc",
            header=None,
            sep=" ",
        )

        ecg_signals = torch.tensor(
            ecg_signals.values
        )  # convert dataframe values to tensor

        ecg_signals = ecg_signals.float()

        ecg_signals = ecg_signals.reshape(-1)

        # Transposing the ecg signals
        # ecg_signals = ecg_signals/6000 # normalization
        ecg_signals = ecg_signals.t()

        # Reshape ecg_signals into a 1D array

        heart_rate = self.heart_rate[index].reshape(-1)

        return ecg_signals, heart_rate

    def __len__(self):
        # return 5000
        return self.ground_truths.shape[0]
