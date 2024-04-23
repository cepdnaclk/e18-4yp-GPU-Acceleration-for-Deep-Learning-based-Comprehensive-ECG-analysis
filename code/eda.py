import torch
import torch.nn as nn
from tqdm import tqdm
import datetime
import wandb
import os
import numpy as np
import random
from sklearn.model_selection import train_test_split
import pandas as pd


from datasets.deepfake_ecg.Deepfake_ECG_Dataset import Deepfake_ECG_Dataset
from datasets.deepfake_ecg.Deepfake_ECG_Dataset import HR_PARAMETER
from datasets.deepfake_ecg.Deepfake_ECG_Dataset import QRS_PARAMETER
from datasets.deepfake_ecg.Deepfake_ECG_Dataset import PR_PARAMETER
from datasets.deepfake_ecg.Deepfake_ECG_Dataset import QT_PARAMETER
import utils.current_server as current_server



parameters = [HR_PARAMETER,QRS_PARAMETER,PR_PARAMETER,QT_PARAMETER]




device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

for parameter in parameters:
    # Create the dataset class
    dataset = Deepfake_ECG_Dataset(parameter=parameter)
    
 
    
    print('Parameter:', parameter)
    print('Length of Dataset:', len(dataset))
    

    null_count = 0
    for data in dataset:
        tensor_data = data[1]
        # print(tensor_data)
        # Check for NaN or null values
        if torch.isnan(tensor_data).any() or tensor_data is None:
            null_count += 1

    print("Null Values Count:", null_count)
    print()
    print()