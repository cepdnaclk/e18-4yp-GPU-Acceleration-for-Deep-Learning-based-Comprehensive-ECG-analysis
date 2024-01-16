import pandas as pd

"""
Get the range of the ground truth values for the deepfake ECG dataset
"""

data_path = "../datasets/deepfake_ecg/filtered_all_normals_121977_ground_truth.csv"
df = pd.read_csv(data_path)

df["avgrrinterval"] = 60 * 1000 / df["avgrrinterval"]
print(df[["avgrrinterval", "pr", "qrs", "qt"]].describe())

"""
Output,

       avgrrinterval             pr            qrs             qt
count  121977.000000  121974.000000  121977.000000  121977.000000
mean       69.779651     158.114123      92.052633     395.206949
std         7.562607      16.848349       8.667925      20.480469
min        59.523810     112.000000      60.000000     318.000000
25%        63.829787     146.000000      86.000000     382.000000
50%        68.181818     158.000000      92.000000     396.000000
75%        73.891626     170.000000      98.000000     410.000000
max       100.671141     208.000000     116.000000     476.000000
"""
