# Vision transformers need the input to be an image.
# So we have to convert the raw ecg data into images and use those RGB values of the images as input to the vision transformer.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# Generate x and y values
x = np.linspace(0, 10, 100)

# Set the desired width and height in pixels
width_pixels = 224
height_pixels = 224

# Calculate DPI based on the desired size
dpi = 100  # You can adjust this value

# Calculate the figure size in inches
fig_width = width_pixels / dpi
fig_height = height_pixels / dpi

# Create a plot with specified size
fig, ax = plt.subplots(figsize=(fig_width, fig_height))
ax.plot(x)
ax.set_title("Example Plot")

# Save the plot as a variable
canvas = FigureCanvas(fig)
canvas.draw()
rgba_array = np.array(canvas.renderer.buffer_rgba())

# Extract RGB values (discard alpha channel)
rgb_array = rgba_array[:, :, :3]

print(rgb_array.shape)


