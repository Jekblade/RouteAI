from PIL import Image, ImageTk
import numpy as np
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from scipy import ndimage
from scipy.spatial import KDTree
from skimage import measure, feature, morphology, filters

# Define main color categories
main_colors = {
    0: "NaN",
    1: "white",
    2: "yellow",
    3: "orange",
    4: "light_green",
    5: "green",
    6: "dark_green",
    7: "light_blue",
    8: "blue",
    9: "olive",
    #--------------
    10: "black",
    11: "brown1",
    12: "brown2",
    13: "brown3",
    14: "brown4",
    15: "brown5",
    16: "purple1",
    17: "purple2",
    18: "pink1",
    19: "pink2"
}

# RGB values for main colors (hand-picked and averaged between various maps)
main_color_values = {
    "NaN": (0, 120, 215),
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
    "brown1": (190, 105, 0),
    "brown2": (190, 105, 40),
    "brown3": (218, 155, 110),
    "brown4": (180, 172, 112),
    "brown5": (130, 140, 75),
    "purple1": (136, 0, 160),
    "purple2": (150, 70, 140),
    "pink1": (215, 0, 120),
    "pink2": (195, 31, 255)
}

# Load the image as NumPy array
def load_image(image_path):
    image = Image.open(image_path)
    image_np = np.array(image)
    return image_np

# Find the closest color using KDTree
def find_closest_color(pixel, main_color_values):
    color_tree = KDTree(list(main_color_values.values()))
    _, index = color_tree.query(pixel[:3])
    closest_color = list(main_color_values.keys())[index]
    return closest_color

# Apply white balance correction
def apply_white_balance(image_np, white_pixel):
  
    if len(white_pixel) == 4:
        white_pixel = white_pixel[:3]  # Take only the RGB values (not alfa)
    
    ideal_white = np.array([255, 255, 255])
    error = ideal_white / white_pixel  # white balance correction factor
    white_balanced_image = np.clip(image_np[:, :, :3] * error, 0, 255).astype(np.uint8)  # Apply to RGB values only
    return white_balanced_image


# Get the white pixel for white balance
def get_white_pixel(image_np):
    root = tk.Tk()
    root.title("Select a White Pixel for White Balance")

    img = Image.fromarray(image_np)
    img_tk = ImageTk.PhotoImage(img)

    label = tk.Label(root, image=img_tk)
    label.pack()

    white_pixel = []

    def click_event(event):
        x, y = event.x, event.y
        white_pixel.extend(image_np[y, x])
        root.quit()

    label.bind('<Button-1>', click_event)
    root.mainloop()

    return np.array(white_pixel)

# Process the image with white balance
def process_image(cropped_map_image, white_pixel, main_color_values):
    map_image_np = np.array(cropped_map_image)
    height, width, _ = np.shape(map_image_np)
    terrain_grid = np.zeros((width, height), dtype=np.uint8)

    # Apply white balance
    white_balanced_image = apply_white_balance(map_image_np, white_pixel)

    # Step 1: Process all pixels
    for y in range(height):
        for x in range(width):
            pixel = white_balanced_image[y, x]
            closest_color = find_closest_color(pixel, main_color_values)
            terrain_grid[x, y] = list(main_colors.keys())[list(main_colors.values()).index(closest_color)]
   
    terrain_grid = np.rot90(terrain_grid, k=1)
    return terrain_grid


# Choose view
def visualize(view_index, terrain_grid, main_colors, main_color_values):
    if view_index == 1:
        visualize_map(terrain_grid, main_colors, main_color_values)
    elif view_index == 2:
        visualize_base_layer(terrain_grid, main_colors, main_color_values)
    elif view_index == 3:
        visualize_lidar_mapping(terrain_grid)
    elif view_index == 4:
        visualize_vertical_lines_removed(terrain_grid)
    elif view_index == 5:
        visualize_trails(terrain_grid)


# VIEW 1 - overall visualization
def visualize_map(terrain_grid, main_colors, main_color_values):

    height, width = np.shape(terrain_grid)
    x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height)) 
    color_rgb_flat = np.array([main_color_values[main_colors[int(color_id)]] for color_id in terrain_grid.flat])

    # preview
    plt.figure(figsize=(8,6)) 
    plt.scatter(x_coords, y_coords, c=color_rgb_flat / 255.0, marker='s')
    plt.title('Map interpretation with Adjusted Colors')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


# VIEW 2 - Base layer
def visualize_base_layer(terrain_grid, main_colors, main_color_values):
    base_layer = np.zeros_like(terrain_grid)
    color_ids = [0, 1, 2, 3, 4, 5, 6, 8] 

    for color_id in color_ids:
        if color_id == 0:
            base_layer[terrain_grid == color_id] = 18 # non-white pixels are set to pink
        else: 
            base_layer[terrain_grid == color_id] = color_id 
    

    # Create a mask to exclude pixels with ID 18 (pink) from the blur operation
    blur_mask = base_layer != 18

    # Apply Gaussian blur only to pixels that are not pink
    base_layer_blurred = np.zeros_like(base_layer, dtype=float)
    base_layer_blurred[blur_mask] = ndimage.gaussian_filter(base_layer[blur_mask].astype(float), sigma=2)
    result_layer = np.where(base_layer == 18, base_layer, base_layer_blurred)

    width, height = result_layer.shape
    x_coords, y_coords = np.meshgrid(np.arange(height), np.arange(width)) 
    color_rgb_flat = np.array([main_color_values[main_colors[int(color_id)]] for color_id in result_layer.flat])
    
    plt.figure(figsize=(8, 6))
    plt.scatter(x_coords, y_coords, c=color_rgb_flat / 255.0, s=1, marker='s')  
    plt.title('Base layer')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


# VIEW 3 - Artificial LIDAR mapping
def visualize_lidar_mapping(terrain_grid):
    lidar_raw = np.copy(terrain_grid)
    brown_ids = [11, 12, 13, 14, 15] 

    lidar = np.isin(lidar_raw, brown_ids)

    coords = np.argwhere(lidar)

    plt.figure(figsize=(8, 6))
    plt.scatter(coords[:, 1], coords[:, 0], c='brown', s=1, marker='s') 
    plt.title('Artificial LIDAR height mapping')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


'''
# VIEW 4 - Removing horizon lines
def visualize_vertical_lines_removed(terrain_grid, main_colors, main_color_values):
    light_blue_id = 6
    blue_id = 7
    black_id = 9
    threshold = 0.4
    width, height = terrain_grid.shape

    colors_to_check = [light_blue_id, blue_id, black_id]

    for x in range(width):
        color_counts = [np.count_nonzero(terrain_grid[x, :] == color_id) for color_id in colors_to_check]

        total_pixels = height
        color_percentages = [count / total_pixels for count in color_counts]

        # replace horizon lines with light_green
        if any(percentage >= threshold for percentage in color_percentages):
            for color_id in colors_to_check:
                terrain_grid[x, terrain_grid[x, :] == color_id] = 3

    x_coords, y_coords = np.meshgrid(np.arange(height), np.arange(width)) 
    color_rgb_flat = np.array([main_color_values[main_colors[color_id]] for row in terrain_grid for color_id in row])

    plt.figure(figsize=(8, 6))
    plt.scatter(x_coords, y_coords, c=color_rgb_flat / 255.0, s=1, marker='s')  
    plt.title('Base layer')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
'''

# VIEW 5 - Connecting trails and paths
def visualize_trails(terrain_grid):
    trails_grid = np.copy(terrain_grid)
    black_id = 9

    mask = np.isin(trails_grid, black_id)
    connected_trails = ndimage.binary_dilation(mask, structure=np.ones((3, 3))).astype(np.uint8)
    trails_smoothed = ndimage.gaussian_filter(connected_trails.astype(float), sigma=1)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(trails_smoothed, cmap='gray')
    plt.title("Connected Trails with Gaussian Blur")
    plt.gca().set_aspect('equal', adjustable="box")
    plt.show()


def main():
 # Select the map image
    file_path = filedialog.askopenfilename(title="Select Map Image", filetypes=[("PNG files", "*.png")])
    
    if not file_path:
        return

    map_image = load_image(file_path)

    # Get the white pixel from the user
    white_pixel = get_white_pixel(map_image)

    # Process the map and save the terrain grid
    terrain_grid = process_image(map_image, white_pixel, main_color_values)

    # Visualize the result
    visualize_map(terrain_grid, main_colors, main_color_values)

    visualize_map(terrain_grid, main_colors, main_color_values)
    visualize_base_layer(terrain_grid, main_colors, main_color_values)
    visualize_lidar_mapping(terrain_grid)
    visualize_vertical_lines_removed(terrain_grid)
    visualize_trails(terrain_grid)

'''
    # Create a dropdown menu for selecting steps
    options = ['Selected map',
                'View 1: Map Interpretation with Adjusted Colors', 
                'View 2: Base Layer Masking', 
                'View 3: Artificial LIDAR Mapping', 
                'View 4: Remove Vertical Lines',
                'View 5: Connecting Roads and Trails']

    def on_select(option):
        selected = selected_option.get()
        print("Selected option:", selected)
        plt.clf()  # Clear the previous plot
        if selected == options[0]:
            # Display the selected map image
            plt.imshow(map_image)
            plt.title("Selected Map")
            plt.gca().set_aspect("equal", adjustable='box')
            plt.show()
        else:
            view_index = int(selected.split()[-1])
            visualize(view_index, terrain_grid, main_colors, main_color_values)

    # Create a Tkinter window
    root = tk.Tk()
    root.title("PNG bitmap conversion")
    root.geometry("400x400")

    # Create a dropdown menu for selecting views
    selected_option = tk.StringVar(root)
    selected_option.set(options[0])  
    dropdown = tk.OptionMenu(root, selected_option, *options, command=on_select)

    
    # Embed the matplotlib plot within the Tkinter window
    canvas = FigureCanvasTkAgg(plt.figure(), master=root)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    # Pack the dropdown menu after the canvas
    dropdown.pack()


    # Add a toolbar for navigating the plot
    toolbar = NavigationToolbar2Tk(canvas, root)
    toolbar.update()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    # Start the Tkinter event loop
    root.mainloop()
    '''



if __name__ == "__main__":
    main()
