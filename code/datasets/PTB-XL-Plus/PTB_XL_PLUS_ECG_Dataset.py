import torch
import pandas as pd
from torch.utils.data import Dataset
import wfdb
import numpy as np
from datetime import datetime

HR_PARAMETER = "hr"
QRS_PARAMETER = "qrs"
PR_PARAMETER = "pr"
QT_PARAMETER = "qt"


class PTB_XL_PLUS_ECGDataset(Dataset):
    def __init__(self, parameter=None):
        super(PTB_XL_PLUS_ECGDataset, self).__init__()

        if parameter not in [HR_PARAMETER, QRS_PARAMETER, PR_PARAMETER, QT_PARAMETER]:
            raise ValueError("Invalid parameter")

        path_to_ptb_xl_dataset = "../../PTB-XL/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/"  # this path might need to be changed if XL dataset is restructured
        ptb_xl_file_names_database_csv = "ptbxl_database.csv"
        path_to_ptb_xl_plus_features = "./features/12sl_features.csv"  # these two also need to be changed if this .py file is taken outsite this folder
        path_to_ptb_xl_plus_statements = "./labels/ptbxl_statements.csv"  # have to check whether this is  12sl_statements  or ptbxl_statements |'scp_codes' vs 'statements'

        self.features_df = pd.read_csv(path_to_ptb_xl_plus_features)
        self.statements_df = pd.read_csv(path_to_ptb_xl_plus_statements)
        self.data_file_names_df = pd.read_csv(
            path_to_ptb_xl_dataset + ptb_xl_file_names_database_csv
        )

        ecg_id_s_rows_to_be_removed_not_normal = []
        for index, row in self.statements_df.iterrows():
            ecg_id = row["ecg_id"]
            scp_codes = row["scp_codes"]

            if "NORM" not in scp_codes:
                ecg_id_s_rows_to_be_removed_not_normal.append(ecg_id)

        rows_to_remove = self.features_df["ecg_id"].isin(
            ecg_id_s_rows_to_be_removed_not_normal
        )
        self.features_df = self.features_df[~rows_to_remove]

        self.features_df.reset_index(drop=True, inplace=True)

        # is it needed to check the directory for existance of files related to filename_hr

        self.X = []
        for index, row in self.features_df.iterrows():
            ecg_id = row["ecg_id"]
            matching_row = self.data_file_names_df[
                self.data_file_names_df["ecg_id"] == ecg_id
            ]
            if not matching_row.empty:
                file_name_hr = matching_row["filename_hr"].values[0]
                file_path = path_to_ptb_xl_dataset + file_name_hr
                data, _ = wfdb.rdsamp(
                    file_path, channels=[0, 1, 6, 7, 8, 9, 10, 11]
                )  # _ meta data from hea file is ignored
                flattened_data = data.flatten()
                # TODO : normalization to be done
                self.X.append(flattened_data)
            else:
                self.features_df.drop(index=index)

        self.X = torch.tensor(
            self.X, dtype=torch.float32
        )  # there is a waring saying this is slow. but we need them as separate tensors so...

        current_time = datetime.now().strftime("%I:%M:%S %p")
        print(f"{current_time} - big stuff read done")

        self.features_df.reset_index(
            drop=True, inplace=True
        )  # in case if there were no matching rows and dropped features from line no 70

        if parameter == HR_PARAMETER:
            self.features_df = self.features_df.dropna(subset=["RR_Mean_Global"])
            hr_torch = torch.tensor(
                self.features_df["RR_Mean_Global"].values, dtype=torch.float32
            )
            # calculate HR
            self.y = 60 * 1000 / hr_torch

        elif parameter == QRS_PARAMETER:
            self.features_df = self.features_df.dropna(subset=["QRS_Dur_Global"])
            self.y = torch.tensor(
                self.features_df["QRS_Dur_Global"].values, dtype=torch.float32
            )

        elif parameter == PR_PARAMETER:
            self.features_df = self.features_df.dropna(subset=["QT_Int_Global"])
            self.y = torch.tensor(
                self.features_df["QT_Int_Global"].values, dtype=torch.float32
            )

        elif parameter == QT_PARAMETER:
            self.features_df = self.features_df.dropna(subset=["PR_Int_Global"])
            self.y = torch.tensor(
                self.features_df["PR_Int_Global"].values, dtype=torch.float32
            )

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        return x, y


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    ## Create an instance of the dataset with the 'hr' parameter
    ecg_dataset = PTB_XL_PLUS_ECGDataset(parameter="hr")

    ## Create a data loader for the dataset
    batch_size = 16
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
