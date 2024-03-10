from PIL import Image, ImageTk
import numpy as np
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import copy
import csv
from scipy import ndimage
from skimage import measure, feature, morphology, filters
import os

# Definint main color categories
main_colors = {
    0: "white",
    1: "yellow",
    2: "orange",
    3: "light_green",
    4: "green",
    5: "dark_green",
    6: "light_blue",
    7: "blue",
    8: "olive",
    #--------------
    9: "black",
    10: "road_orange",
    11: "brown1",
    12: "brown2",
    13: "brown3",
    14: "brown4",
    15: "purple1",
    16: "purple2",
    17: "pink1",
    18: "pink2"
}

# RGB values for main colors (hand-picked and averaged between various maps)
main_color_values = {
    "white": (255, 255, 255),
    "yellow": (250, 190, 75),
    "orange": (253, 217, 148),
    "light_green": (196, 230, 190),
    "green": (140, 205, 130),
    "dark_green": (40, 170, 80),
    "light_blue": (120, 220, 230),
    "blue": (0, 160, 215),
    "olive": (160, 158, 58),
    #------------------------
    "black": (40, 40, 40),
    "road_orange": (228, 170, 120),
    "brown1": (190, 105, 0),
    "brown2": (190, 105, 40),
    "brown3": (218, 155, 110),
    "brown4": (180, 172, 112),
    "purple1": (136, 0, 160),
    "purple2": (150, 70, 140),
    "pink1": (215, 0, 120),
    "pink2": (195, 31, 255)
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

def process_image(cropped_map_image):
    map_image_np = np.array(cropped_map_image)
    height, width, _ = map_image_np.shape
    terrain_grid = np.zeros((width, height), dtype=np.uint8)

    # Step 1: Process all pixels
    for y in range(height):
        for x in range(width):
            pixel = map_image_np[y, x]
            closest_color = find_closest_color(pixel)
            terrain_grid[x, y] = list(main_colors.keys())[list(main_colors.values()).index(closest_color)]
    
    terrain_grid = np.rot90(terrain_grid, k=1)
    width, height = np.shape(terrain_grid)
    x_coords, y_coords = np.meshgrid(np.arange(height), np.arange(width)) 
    color_rgb_flat = np.array([main_color_values[main_colors[color_id]] for row in terrain_grid for color_id in row])

    # Create a scatter plot of the pixels with their respective main color values
    plt.figure(figsize=(8,6)) 
    plt.scatter(x_coords, y_coords, c=color_rgb_flat / 255.0, marker='s')
    plt.title('Map interpretation with Adjusted Colors')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
    terrain_grid = np.rot90(terrain_grid, k=-1)



    # -=-=-=-=-=-=-=-
    # Step 2: Artificial LIDAR height mapping

    lidar = np.copy(terrain_grid)
    brown_ids = [11, 12, 13, 14] 
    other_ids = [9, 15, 16, 17, 18]

    # Create masks for brown and for other pixels that should be filled with brown
    brown_mask = np.isin(lidar, brown_ids)
    replace_mask = np.isin(lidar, other_ids)

    dilated_brown_mask = ndimage.binary_dilation(brown_mask, structure=np.ones((3,3)))
    black_adjacent_to_brown = np.logical_and(replace_mask, dilated_brown_mask)

    # Integrates black pixels adjacent to brown into the brown LIDAR grid
    combined_mask = np.logical_or(brown_mask, black_adjacent_to_brown)

    lidar[~combined_mask] = 0  # Other pixels to white
    combined_mask = np.rot90(combined_mask, k=1)

    coords = np.argwhere(combined_mask)

    plt.figure(figsize=(8, 6))
    plt.scatter(coords[:, 1], coords[:, 0], c='brown', s=1, marker='s')  # Plot points with id 9 in brown color
    plt.title('Artificial LIDAR with Connected Missing Ends')
    plt.colorbar(label='Region Label')
    plt.gca().set_aspect('equal', adjustable='box')


    # -=-=-=-=-=-=-=-
    # Step 3: Check for vertical lines and remove them (magnetic horizon lines)

    light_blue_id = 6
    blue_id = 7
    black_id = 9
    threshold = 0.3

    colors_to_check = [light_blue_id, blue_id, black_id]

    for x in range(width):
        color_counts = [np.count_nonzero(terrain_grid[x, :] == color_id) for color_id in colors_to_check]
        print(color_counts)

        total_pixels = height
        color_percentages = [count / total_pixels for count in color_counts]
        print(color_percentages)

        # replace horizon line with light_green
        if any(percentage >= threshold for percentage in color_percentages):
            for color_id in colors_to_check:
                terrain_grid[x, terrain_grid[x, :] == color_id] = 3


   # -=-=-=-=-=-=-=-
   # Step 4: Black pixel classification - connecting roads and trails

    trails_grid = np.copy(terrain_grid)
    black_id = 9

    # Create a mask for black color
    mask = np.isin(trails_grid, black_id)
    trails_grid[~mask] = 0
    mask = morphology.remove_small_objects(mask, min_size=4)
    mask = np.rot90(mask, k=1)

    # Detect paths:
    edges = feature.canny(mask, sigma=0.5)

    # Close gaps between disconnected paths
    trails_connected = morphology.binary_closing(edges, morphology.disk(4))

    skeleton = morphology.skeletonize(trails_connected)

    # Calculate the distance transform of the skeleton image
    distance_transform = filters.sobel(skeleton)

    # Identify thickest parts of roads based on distance transform
    thickest_parts = distance_transform >= 1
    distance_1 = morphology.binary_dilation(skeleton) ^ skeleton

    trails = thickest_parts | distance_1 | skeleton

    coords = np.argwhere(trails)
    plt.figure(figsize=(8, 6))
    plt.scatter(coords[:, 1], coords[:, 0], c='black', s=1, marker='s')  # Plot points with id 9 in brown color
    plt.title('Trails connect')
    plt.colorbar(label='Region Label')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

    return trails


def main():
    # Select map image
    file_path = filedialog.askopenfilename(title="Select Map Image", filetypes=[("PNG files", "*.png")])
    
    if file_path:
        map_image = Image.open(file_path)

        # Process the map and save the terrain grid
        terrain_colors = process_image(map_image)

        width, height = np.shape(terrain_colors)
        x_coords, y_coords = np.meshgrid(np.arange(height), np.arange(width)) 

        # Flatten the arrays for plotting
        x_flat = x_coords.flatten()
        y_flat = y_coords.flatten()
        color_rgb_flat = np.array([main_color_values[main_colors[color_id]] for row in terrain_colors for color_id in row])

        # Create a scatter plot of the pixels with their respective main color values
        plt.figure(figsize=(width/100, height/100)) 
        plt.scatter(x_flat, y_flat, c=color_rgb_flat / 255.0, marker='s')
        plt.title('Map interpretation with Adjusted Colors')
        plt.gca().set_aspect('equal', adjustable='box')  # Set equal aspect ratio

if __name__ == "__main__":
    main()

