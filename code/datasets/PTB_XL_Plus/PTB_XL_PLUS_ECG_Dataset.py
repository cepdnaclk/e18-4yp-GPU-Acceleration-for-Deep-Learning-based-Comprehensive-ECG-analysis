import torch
import pandas as pd
from torch.utils.data import Dataset
import wfdb
import numpy as np
from datetime import datetime
from tqdm import tqdm
import socket
import os
import utils.datasets as utils_datasets

HR_PARAMETER = "hr"
QRS_PARAMETER = "qrs"
PR_PARAMETER = "pr"
QT_PARAMETER = "qt"


class PTB_XL_PLUS_ECGDataset(Dataset):
    def __init__(self, parameter=None, num_of_leads=8):
        super(PTB_XL_PLUS_ECGDataset, self).__init__()
        
        self.num_of_leads = num_of_leads

        if parameter not in [HR_PARAMETER, QRS_PARAMETER, PR_PARAMETER, QT_PARAMETER]:
            raise ValueError("Invalid parameter")

        hostname = socket.gethostname()
        if hostname == "ampere":
            print("Running in ampere server. Checking if data is available in ram")
            path_to_ptb_xl_dataset = "/dev/shm/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/"
            path_to_ptb_xl_plus_features = "/dev/shm/ptb-xl-a-comprehensive-electrocardiographic-feature-dataset-1.0.1/features/12sl_features.csv"
            path_to_ptb_xl_plus_statements = "/dev/shm/ptb-xl-a-comprehensive-electrocardiographic-feature-dataset-1.0.1/labels/ptbxl_statements.csv"

            if os.path.exists(path_to_ptb_xl_dataset):
                print("PTB-XL found in RAM. Continuing")
            else:
                utils_datasets.download_and_extract_ptb_xl_dataset_to_ram()

            if os.path.exists(path_to_ptb_xl_plus_features):
                print("PTB-XL+ found in RAM. Continuing")
            else:
                utils_datasets.download_and_extract_ptb_xl_plus_dataset_to_ram()
        else:
            print("Not running in ampere. Loading data from disk")
            path_to_ptb_xl_dataset = "./datasets/PTB_XL/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/"
            path_to_ptb_xl_plus_features = "./datasets/PTB_XL_Plus/ptb-xl-a-comprehensive-electrocardiographic-feature-dataset-1.0.1/features/12sl_features.csv"
            path_to_ptb_xl_plus_statements = "./datasets/PTB_XL_Plus/ptb-xl-a-comprehensive-electrocardiographic-feature-dataset-1.0.1/labels/ptbxl_statements.csv"

        ptb_xl_file_names_database_csv = "ptbxl_database.csv"

        self.features_df = pd.read_csv(path_to_ptb_xl_plus_features)
        self.statements_df = pd.read_csv(path_to_ptb_xl_plus_statements)
        self.data_file_names_df = pd.read_csv(path_to_ptb_xl_dataset + ptb_xl_file_names_database_csv)

        ecg_id_s_rows_to_be_removed_not_normal = []
        for index, row in self.statements_df.iterrows():
            ecg_id = row["ecg_id"]
            scp_codes = row["scp_codes"]

            if "NORM" not in scp_codes:
                ecg_id_s_rows_to_be_removed_not_normal.append(ecg_id)

        rows_to_remove = self.features_df["ecg_id"].isin(ecg_id_s_rows_to_be_removed_not_normal)
        self.features_df = self.features_df[~rows_to_remove]

        self.features_df.reset_index(drop=True, inplace=True)

        # is it needed to check the directory for existance of files related to filename_hr

        self.X = []
        for index, row in tqdm(
            self.features_df.iterrows(),
            total=len(self.features_df),
            desc="Reading data files",
        ):
            ecg_id = row["ecg_id"]
            matching_row = self.data_file_names_df[self.data_file_names_df["ecg_id"] == ecg_id]
            if not matching_row.empty:
                file_name_hr = matching_row["filename_hr"].values[0]
                file_path = path_to_ptb_xl_dataset + file_name_hr
                if num_of_leads == 12:
                    data, _ = wfdb.rdsamp(file_path, channels=[0, 1, 2, 3, 4, 5,6, 7, 8, 9, 10, 11])  # _ meta data from hea file is ignored
                elif num_of_leads == 8:
                    data, _ = wfdb.rdsamp(file_path, channels=[0, 1, 6, 7, 8, 9, 10, 11])  # _ meta data from hea file is ignored
                else:
                    raise Exception("number of leads is wrong. It should be either 8 or 12")
                # flattened_data = data.flatten()

                # # normalization
                # flattened_data = flattened_data / 27

                # flattened_data = torch.tensor(flattened_data, dtype=torch.float32)
                data = torch.Tensor(data).transpose(0, 1)  # Transpose the tensor to (8, 5000)
                self.X.append(data)
            else:
                self.features_df.drop(index=index)

        self.features_df.reset_index(drop=True, inplace=True)  # in case if there were no matching rows and dropped features from line no 70

        if parameter == HR_PARAMETER:
            rr_mean_global_series = self.features_df["RR_Mean_Global"]
            hr_torch = torch.tensor(rr_mean_global_series.values, dtype=torch.float32)
            # Calculate HR
            self.y = 60 * 1000 / hr_torch

        if parameter == QRS_PARAMETER:
            qrs_dur_series = self.features_df["QRS_Dur_Global"]
            self.y = torch.tensor(qrs_dur_series.values, dtype=torch.float32)

        elif parameter == PR_PARAMETER:
            qt_int_series = self.features_df["QT_Int_Global"]
            self.y = torch.tensor(qt_int_series.values, dtype=torch.float32)

        elif parameter == QT_PARAMETER:
            pr_int_series = self.features_df["PR_Int_Global"]
            self.y = torch.tensor(pr_int_series.values, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        return x, y.reshape(-1)


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    ## Create an instance of the dataset with the 'hr' parameter
    ecg_dataset = PTB_XL_PLUS_ECGDataset(parameter="hr")

    ## Create a data loader for the dataset
    batch_size = 16
    data_loader = DataLoader(ecg_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

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
