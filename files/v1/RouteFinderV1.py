import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load the map image
map_image = Image.open("map.png")

# Get the width and height of the image
width, height = map_image.size

# Create arrays to store pixel coordinates and colors
x_coords = []
y_coords = []
colors = []

# Iterate over each pixel in the image
for y in range(height):
    for x in range(width):
        # Get the RGB color of the pixel
        pixel_color = map_image.getpixel((x, y))

        # Append the pixel coordinates and color to the respective arrays
        x_coords.append(x)
        y_coords.append(height - y)  # Invert y-axis to match matplotlib coordinate system
        colors.append(pixel_color)

# Convert lists to numpy arrays for efficient processing
x_coords = np.array(x_coords)
y_coords = np.array(y_coords)
colors = np.array(colors) / 255.0  # Normalize color values to range [0, 1]

# Plot the pixels with their respective colors
plt.figure(figsize=(width/100, height/100))  # Adjust figure size based on image dimensions
plt.scatter(x_coords, y_coords, color=colors, s=1)  # Set marker size to 1 for single pixel effect
plt.gca().invert_yaxis()  # Invert y-axis to match the image orientation
plt.show()
