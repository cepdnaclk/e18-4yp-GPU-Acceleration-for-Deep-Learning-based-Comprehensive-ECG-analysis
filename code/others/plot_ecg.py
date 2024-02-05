import numpy as np
import matplotlib.pyplot as plt

# Strip the first 300 and last 200 rows
# Load the data from the file
file_path = "/content/drive/MyDrive/0.asc"

data = np.loadtxt(file_path)
ecg_data = data[300:-200]

# Reshape the data into a 3x3 grid
num_rows = 3
num_cols = 3
grid_data = ecg_data.reshape(num_rows, num_cols, -1)

# Create a figure with subplots
fig, axs = plt.subplots(num_rows, num_cols, figsize=(9, 9))
fig.subplots_adjust(hspace=0, wspace=0)

# Plot each waveform in grayscale
for i in range(num_rows):
    for j in range(num_cols):
        axs[i, j].plot(grid_data[i, j], color='black')
        axs[i, j].axis('off')  # Remove axes
        axs[i, j].set_xticks([])
        axs[i, j].set_yticks([])

# Convert the grid data to grayscale (average of RGB channels)
# gray_grid_data = np.mean(grid_data, axis=2)
gray_grid_data = grid_data

# Print the grayscale values as a 2D matrix
print("Grayscale matrix:")
print(gray_grid_data)
print("Shape : ", gray_grid_data.shape)

# Save the resulting image
plt.savefig('ecg_grid.png', dpi=256, bbox_inches='tight')
plt.show()
plt.close()

print("ECG waveform grid saved as 'ecg_grid.png'")
