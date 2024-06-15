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
from sklearn.model_selection import train_test_split
import ast


HR_PARAMETER = "hr"
QRS_PARAMETER = "qrs"
PR_PARAMETER = "pr"
QT_PARAMETER = "qt"

SUB_DATASET_A = "A"
SUB_DATASET_B = "B"


class PTB_XL_PLUS_ECGDataset(Dataset):
    labels = ["MI", "STTC", "HYP", "NORM", "CD"]

    def __init__(self, parameter=None, num_of_leads=8, sub_dataset=None, is_classification=False):
        super(PTB_XL_PLUS_ECGDataset, self).__init__()

        self.num_of_leads = num_of_leads
        self.is_classification = is_classification
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

        # ecg_id_s_rows_to_be_removed_not_normal = []
        # for index, row in self.statements_df.iterrows():
        #     ecg_id = row["ecg_id"]
        #     scp_codes = row["scp_codes"]

        #     if "NORM" not in scp_codes:
        #         ecg_id_s_rows_to_be_removed_not_normal.append(ecg_id)

        # rows_to_remove = self.features_df["ecg_id"].isin(ecg_id_s_rows_to_be_removed_not_normal)
        # self.features_df = self.features_df[~rows_to_remove]

        # self.features_df.reset_index(drop=True, inplace=True)

        # drop NaN records
        columns_to_check = ["RR_Mean_Global", "QRS_Dur_Global", "QT_Int_Global", "PR_Int_Global"]

        # check and drop if these columns have NaN
        self.features_df = self.features_df.dropna(subset=columns_to_check)

        # is it needed to check the directory for existance of files related to filename_hr

        self.statements_df.scp_codes = self.statements_df.scp_codes.apply(lambda x: ast.literal_eval(x))

        # normalize self.X
        # self.X = (self.X - np.min(self.X)) / (np.max(self.X) - np.min(self.X))

        # Load scp_statements.csv for diagnostic aggregation
        self.agg_df = pd.read_csv(path_to_ptb_xl_dataset + "scp_statements.csv", index_col=0)
        self.agg_df = self.agg_df[self.agg_df.diagnostic == 1]

        # Apply diagnostic superclass and add the 'diagnostic_superclass' column

        self.statements_df["diagnostic_superclass"] = self.statements_df.scp_codes.apply(self.aggregate_diagnostic)
        self.y = self.statements_df[self.statements_df["diagnostic_superclass"].apply(lambda x: len(x) == 1)]

        # iterate throught self.features_df and get the 'ecg_id', remove it from self.features_df if it is not in self.y
        self.features_df = self.features_df[self.features_df["ecg_id"].isin(self.y["ecg_id"])]
        self.y = self.y[self.y["ecg_id"].isin(self.features_df["ecg_id"])]
        
        if not is_classification:

            if parameter not in [HR_PARAMETER, QRS_PARAMETER, PR_PARAMETER, QT_PARAMETER]:
                raise ValueError("Invalid parameter")

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

        self.X = []
        if is_classification:
            iterate_over = self.y
        else:
            iterate_over = self.features_df
            
        for index, row in tqdm(
            iterate_over.iterrows(),
            total=len(iterate_over),
            desc="Reading data files",
        ):
            ecg_id = row["ecg_id"]
            matching_row = self.data_file_names_df[self.data_file_names_df["ecg_id"] == ecg_id]
            if not matching_row.empty:
                file_name_hr = matching_row["filename_hr"].values[0]
                file_path = path_to_ptb_xl_dataset + file_name_hr
                if num_of_leads == 12:
                    data, _ = wfdb.rdsamp(file_path, channels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])  # _ meta data from hea file is ignored
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
                self.y.drop(index=index)

        # self.y.reset_index(drop=True, inplace=True)

        # select the subset
        if sub_dataset == None:
            print("No sub dataset specified. Using the entire dataset")
        else:
            subsetA, subsetB = train_test_split(range(len(self.X)), test_size=0.5, random_state=42, shuffle=True)
            assert len(self.X) == len(self.y), "X and y should have the same length"
            if sub_dataset == SUB_DATASET_A:
                self.X = [self.X[i] for i in subsetA]
                if is_classification:
                    self.y = [self.y.iloc[i] for i in subsetA]
                else:
                    self.y = self.y[subsetA]
            elif sub_dataset == SUB_DATASET_B:
                self.X = [self.X[i] for i in subsetB]
                if is_classification:
                    self.y = [self.y.iloc[i] for i in subsetB]
                else:
                    self.y = self.y[subsetB]
            else:
                raise Exception("Invalid sub dataset. It should be either A or B")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]

        if self.is_classification:
            y = self.y[idx]["diagnostic_superclass"][0]
            y = torch.tensor([y == i for i in self.labels], dtype=torch.float32)
        else:
            y = self.y[idx].reshape(-1)

        return x, y

    def aggregate_diagnostic(self, y_list):
        tmp = []
        for each_tuple in y_list:
            each_tuple = each_tuple[0]
            if each_tuple in self.agg_df.index:
                tmp.append(self.agg_df.loc[each_tuple].diagnostic_class)
        return list(set(tmp))


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
