from PIL import Image, ImageTk
import numpy as np
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import copy
import csv
import os

# Define main colors for categories
main_colors = {
    0: "white",
    1: "light_green",
    2: "green",
    3: "dark_green",
    4: "orange",
    5: "black",
    6: "yellow",
    7: "blue",
    8: "olive",
    9: "brown",
    10: "dark_brown",
    11: "purple",
    12: "pink"
}

# RGB values for main colors
main_color_values = {
    "white": (255, 255, 255),
    "light_green": (204, 224, 191),
    "green": (150, 200, 150),
    "dark_green": (85, 165, 95),
    "orange": (245, 210, 150),
    "black": (50, 50, 50),
    "yellow": (250, 200, 90),
    "blue": (80, 180, 220),
    "olive": (150, 150, 50),
    "brown": (210, 175, 150),
    "dark_brown": (170, 80, 30),
    "purple": (130, 20, 115),
    "pink": (225, 125, 172)
}

def find_closest_color(pixel):
    min_distance = float('inf')
    closest_color = None
    for color, rgb_value in main_color_values.items():
        distance = color_distance(pixel, rgb_value)
        if distance < min_distance:
            min_distance = distance
            closest_color = color
    return closest_color

def color_distance(color1, color2):
    return sum((c1 - c2) ** 2 for c1, c2 in zip(color1, color2))

def process_image(map_image, file_name):
    map_image_np = np.array(map_image)
    height, width, _ = map_image_np.shape
    terrain_grid = np.zeros((width, height), dtype=np.uint8)

    # Step 1: Process all pixels
    for y in range(height):
        for x in range(width):
            pixel = map_image_np[y, x]
            closest_color = find_closest_color(pixel)
            terrain_grid[x, y] = list(main_colors.keys())[list(main_colors.values()).index(closest_color)]
    
    # Step 2: Check for vertical lines and remove them (magnetic horizon lines)
    black_id = 5
    olive_id = 6
    blue_id = 7
    threshold = 0.6
    columns_to_remove = []

    colors_to_check = [black_id, olive_id, blue_id]

    for x in range(width):
        color_counts = [np.count_nonzero(terrain_grid[x, :] == color_id) for color_id in colors_to_check]
        total_pixels = height

        color_percentages = [count / total_pixels for count in color_counts]

        if any(percentage >= threshold for percentage in color_percentages):
            terrain_grid = np.delete(terrain_grid, columns_to_remove, axis=0)

   # Step 3: Black pixel classification - connecting roads and trails
    terrain_grid_final = np.copy(terrain_grid)

    for y in range(0, height):
        for x in range(0, width):
            if terrain_grid[x, y] == black_id:
                max_connectivity = 0
                best_direction = (0, 0)
                
                # Analyze the 5x5 grid around the black pixel
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        # Skip the current pixel
                        if dx == 0 and dy == 0:
                            continue

                        # Count connectivity in the current direction
                        connectivity = 0
                        for i in range(-1, 2):  # Check one more pixel in each direction
                            px, py = x + i * dx, y + i * dy
                            if 0 <= px < width and 0 <= py < height and terrain_grid[px, py] == black_id:
                                connectivity += 1

                        if connectivity > max_connectivity:
                            max_connectivity = connectivity
                            best_direction = (dx, dy)

                # Mark pixels based on the best direction
                if best_direction[0] == 0 or best_direction[1] == 0:  # Horizontal or vertical direction
                    for i in range(-2, 3):
                        for j in range(-2, 3):
                            px, py = x + i * best_direction[0], y + j * best_direction[1]
                            if 0 <= px < width and 0 <= py < height and terrain_grid[px, py] != black_id:
                                terrain_grid_final[px, py] = black_id
                else:  # Diagonal direction
                    for i in range(-1, 2):
                        for j in range(-1, 2):
                            # Skip marking the corners based on the best diagonal direction
                            if (i * best_direction[0] == -1 and j * best_direction[1] == -1) or \
                            (i * best_direction[0] == -1 and j * best_direction[1] == 1) or \
                            (i * best_direction[0] == 1 and j * best_direction[1] == -1) or \
                            (i * best_direction[0] == 1 and j * best_direction[1] == 1):
                                continue

                            px, py = x + i, y + j
                            if 0 <= px < width and 0 <= py < height and terrain_grid[px, py] != black_id:
                                terrain_grid_final[px, py] = black_id
    
    # Save the terrain_grid
    file_path = os.path.join("files/terrain_grids", f"{file_name}.txt")
    np.savetxt(file_path, terrain_grid_final, fmt='%d')

    return terrain_grid_final


def main():
    # Select map image
    file_path = filedialog.askopenfilename(title="Select Map Image", filetypes=[("PNG files", "*.png")])
    
    if file_path:
        map_image = Image.open(file_path)
       

        # Extract the file name from the file path
        file_name = os.path.splitext(os.path.basename(file_path))[0]  # Remove the file extension

        terrain_colors = np.array([])

        terrain_colors = process_image(map_image, file_name)

        width, height = np.shape(terrain_colors)
        x_coords, y_coords = np.meshgrid(np.arange(height), np.arange(width)) 

        # Flatten the arrays for plotting
        x_flat = x_coords.flatten()
        y_flat = y_coords.flatten()
        color_rgb_flat = np.array([main_color_values[main_colors[color_id]] for row in terrain_colors for color_id in row])

        # Create a scatter plot of the pixels with their respective main color values
        plt.figure(figsize=(width / 100, height / 100)) 
        plt.scatter(x_flat, y_flat, c=color_rgb_flat / 255.0, marker='s')
        plt.title('Map interpretation with Adjusted Color Categories')

        plt.gca().set_aspect('equal', adjustable='box')  # Set equal aspect ratio
        plt.show()

if __name__ == "__main__":
    main()

