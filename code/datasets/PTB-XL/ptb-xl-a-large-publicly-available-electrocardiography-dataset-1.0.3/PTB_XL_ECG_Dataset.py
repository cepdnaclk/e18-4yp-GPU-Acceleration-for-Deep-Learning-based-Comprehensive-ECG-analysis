import torch
import pandas as pd
from torch.utils.data import Dataset
import ast
import wfdb
import numpy as np

class ECGDataset(Dataset):
    def __init__(self, path='./', sampling_rate=100):
        self.path = path
        self.sampling_rate = sampling_rate

        # Load and convert annotation data
        self.Y = pd.read_csv(path + 'ptbxl_database.csv', index_col='ecg_id')
        self.Y.scp_codes = self.Y.scp_codes.apply(lambda x: ast.literal_eval(x))

        # Load raw signal data
        self.X = self.load_raw_data()

        # Load scp_statements.csv for diagnostic aggregation
        self.agg_df = pd.read_csv(path + 'scp_statements.csv', index_col=0)
        self.agg_df = self.agg_df[self.agg_df.diagnostic == 1]
        
        # Apply diagnostic superclass and add the 'diagnostic_superclass' column
        
        self.Y['diagnostic_superclass'] = self.Y.scp_codes.apply(self.aggregate_diagnostic)


    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        x = torch.Tensor(self.X[idx])  # Assuming X is a NumPy array
        diagnostic_superclass = self.Y['diagnostic_superclass'].iloc[idx]
        y = torch.tensor([1 if label in diagnostic_superclass else 0 for label in ["NORM"]], dtype=torch.float32)
        # y = torch.tensor(self.Y['diagnostic_superclass'].iloc[idx], dtype=torch.long) # this is a text as NORM... (normal or not)

        return x, y

    def load_raw_data(self):
        if self.sampling_rate == 100:
            data = [wfdb.rdsamp(self.path + f, channels=[0, 1, 6, 7, 8, 9, 10, 11]) for f in self.Y.filename_lr]
        else:
            data = [wfdb.rdsamp(self.path + f, channels=[0, 1, 6, 7, 8, 9, 10, 11]) for f in self.Y.filename_hr]

        data = np.array([signal for signal, _ in data]) # here _ is metadata : leftout
        return data

    def aggregate_diagnostic(self, y_dic):
        tmp = []
        for key in y_dic.keys():
            if key in self.agg_df.index:
                tmp.append(self.agg_df.loc[key].diagnostic_class)
        return list(set(tmp))


# Example usage: if the path is different and the sampling rate different 
# default set to current directory and sampleRate 100
# path = './'
# sampling_rate = 100
# ecg_dataset = ECGDataset(path, sampling_rate)

# ecg_dataset = ECGDataset()