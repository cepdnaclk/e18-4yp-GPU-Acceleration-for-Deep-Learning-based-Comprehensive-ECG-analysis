import torch
import pandas

HR_PARAMETER = "hr"
QRS_PARAMETER = "qrs"
PR_PARAMETER = "pr"
QT_PARAMETER = "qt"


class Deepfake_ECG_Dataset(torch.utils.data.Dataset):
    """
    Deepfake ECG dataset filtered to only include normal ECGs
    Contains 121977 ECGs
    ECG signals are returned as a 1D tensor (40k numbers)
    Parameters are returned as a 1D tensor
    """

    def __init__(self, parameter=None):
        super(Deepfake_ECG_Dataset, self).__init__()

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

            # Convert dataframe values to tensor
            ecg_signals = torch.tensor(ecg_signals.values)
            ecg_signals = ecg_signals.float()
            ecg_signals = ecg_signals.reshape(-1)

            # Transposing the ECG signals
            ecg_signals = ecg_signals / 3500  # normalization
            ecg_signals = ecg_signals.t()

            # Store the loaded ASC file in the dictionary
            self.loaded_asc_files[filename] = ecg_signals

        parameter = self.parameter[index].reshape(-1)

        return ecg_signals, parameter

    def __len__(self):
        # return 1000
        return self.ground_truths.shape[0]
