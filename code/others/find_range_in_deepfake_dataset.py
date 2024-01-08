import os
from tqdm import tqdm

"""
Find the range of the deepfake dataset in order to normalize it
"""

data_path = (
    "../datasets/deepfake_ecg/from_006_chck_2500_150k_filtered_all_normals_121977/"
)
files = os.listdir(data_path)

minValue = float("inf")
maxValue = float("-inf")

for file_name in tqdm(files):
    file_path = os.path.join(data_path, file_name)

    try:
        with open(file_path, "r") as f:
            numbers = [float(number) for line in f for number in line.split()]
            maxValue = max(maxValue, max(numbers))
            minValue = min(minValue, min(numbers))

    except Exception as e:
        print(f"Error reading {file_name}: {str(e)}")

print(f"Min: {minValue}")
print(f"Max: {maxValue}")
