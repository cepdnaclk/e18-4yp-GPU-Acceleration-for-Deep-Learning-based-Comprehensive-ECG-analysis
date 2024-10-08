import os
from tqdm import tqdm

"""
Deepfake ECG Dataset
Check if all the asc files have 40k numbers in them
"""

data_path = '../datasets/deepfake_ecg/from_006_chck_2500_150k_filtered_all_normals_121977/'
files = os.listdir(data_path)

for file_name in tqdm(files):
    file_path = os.path.join(data_path, file_name)
    
    try:
        with open(file_path, 'r') as f:
            number_count = sum(len(line.split()) for line in f)
            
        if number_count != 40_000:
            print(f'{file_name} has {number_count} numbers')
    
    except Exception as e:
        print(f'Error reading {file_name}: {str(e)}')
