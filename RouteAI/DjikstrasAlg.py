from PIL import Image, ImageTk
import numpy as np
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import networkx as nx
from scipy import ndimage
from skimage import morphology
import time

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
    #--------------
    8: "black",
    9: "brown1",
    10: "brown2",
    11: "brown3",
    12: "brown4",
    13: "brown5",
    14: "purple1",
    15: "purple2",
    16: "pink1",
    17: "pink2"
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

# Costs associated with terrain runnability
color_costs = {
    "white": 1.5, 
    "yellow": 1,
    "orange": 1.2,
    "light_green": 3,
    "green": 5,
    "dark_green": 8,
    "light_blue": 3,
    "blue": 3,

    #------------------------
    "black": 0.05,
    "brown1": 10,
    "brown2": 10,
    "brown3": 10,
    "brown4": 10,
    "brown5": 10,
    "purple1": 1.5,
    "purple2": 1.5,
    "pink1": 1.5,
    "pink2": 1.5,
}

class PointSelectionApp:
    def __init__(self, master, map_image):
        self.master = master
        self.master.title("Selection of start point and end point")

        self.points = []
        self.start_point_selected = False
        self.finish_point_selected = False

        self.canvas = tk.Canvas(master)
        self.canvas.pack(side="left", fill="y")

        #Scrollbars
        self.v_scrollbar = tk.Scrollbar(master, orient="vertical", command=self.canvas.yview)
        self.v_scrollbar.pack(side="right", fill="y")
        self.canvas.configure(yscrollcommand=self.v_scrollbar.set)

        self.map_photo = ImageTk.PhotoImage(map_image)
        self.image_item = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.map_photo)
        canvas_width = self.map_photo.width()
        canvas_height = self.map_photo.height()
        self.canvas.config(width=canvas_width, height=canvas_height)

        self.canvas.update_idletasks()
        self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))

        # Actions
        self.canvas.bind("<Button-1>", self.add_point)
        self.master.bind("<Return>", self.complete_selection)


    def add_point(self, event):
        x, y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)

        if len(self.points) <= 2:
           
            if not self.start_point_selected:
                # Remove previously selected start point if exists
                if len(self.points) > 0:
                    self.canvas.delete(self.points[-1][2])

                # Draw the points
                start_point_item = self.canvas.create_oval(x - 10, y - 10, x + 10, y + 10, outline="red", width=2, fill='')
                self.points.append((x, y, start_point_item))
                self.start_point_selected = True

            # Do the same for the finish point
            elif self.start_point_selected and not self.finish_point_selected:

                finish_point_item = self.canvas.create_oval(x - 10, y - 10, x + 10, y + 10, outline="blue", width=2, fill='')
                self.points.append((x, y, finish_point_item))
                self.finish_point_selected = True

    def complete_selection(self, event):
        if self.start_point_selected and self.finish_point_selected:
            self.master.quit()


def crop_map_around_points(map_image, raw_start_point, raw_end_point, buffer_size=100):
    # Extract coordinates
    start_x, start_y = raw_start_point
    end_x, end_y = raw_end_point

    map_width, map_height = map_image.size

    # Determine the bounding box for cropping
    min_x = max(0, min(start_x, end_x) - buffer_size)
    max_x = min(map_width, max(start_x, end_x) + buffer_size)
    min_y = max(0, min(start_y, end_y) - buffer_size)
    max_y = min(map_height, max(start_y, end_y) + buffer_size)

    cropped_image = map_image.crop((min_x, min_y, max_x, max_y))

    # Update the adjusted start and end points
    start_point = (start_x - min_x, start_y - min_y)
    end_point = (end_x - min_x, end_y - min_y)

    return cropped_image, start_point, end_point



# -=-=-=-=-=-=-=-=-=-
# Step 0: Process all pixels 
def process_image(cropped_map_image):
    seconds = time.time()
    progress = 0

    map_image_np = np.array(cropped_map_image)
    height, width, _ = map_image_np.shape
    raw_terrain_grid = np.zeros((height, width), dtype=np.uint8)

    for y in range(height):
        progress = int((y / height) * 100)
        print(f"   Image processing: {round(progress,1)}%", end='\r')

        for x in range(width):
            pixel = map_image_np[y, x]
            closest_color = find_closest_color(pixel)
            raw_terrain_grid[y, x] = list(main_colors.keys())[list(main_colors.values()).index(closest_color)]
    print(f"Time elapsed in image processing: {round((time.time() - seconds), 3)} s")
    return raw_terrain_grid

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


# -=-=-=-=-=-=-=-=-=-
# Step 2: Give more weight to paths and trails
def coonnect_trails(raw_terrain_grid):
    seconds = time.time()
    black_id = 8
    mask = np.isin(raw_terrain_grid, black_id)

    connected_trails = ndimage.binary_dilation(mask, structure=np.ones((2, 2))).astype(np.uint8)

    raw_terrain_grid[connected_trails == 1] = black_id

    print(f"Connecting roads: {round((time.time() - seconds), 3)} s")
    return raw_terrain_grid


# -=-=-=-=-=-=-=-=-=-
# Step 3: Artificial LIDAR terrain mapping
def artificial_lidar(raw_terrain_grid):

    lidar = np.copy(raw_terrain_grid)
    brown_ids = [9, 10, 11, 12, 13] 
    other_ids = [8, 14, 15, 16, 17]

    # Create masks for brown and for other pixels that should be filled with brown
    brown_mask = np.isin(lidar, brown_ids)
    relief_coords = np.argwhere(brown_mask)

    return relief_coords


# -=-=-=-=-=-=-=-
# Step 4: Remove horizon lines if black
def horizon_lines(terrain_grid):
    seconds = time.time()
    width, height = terrain_grid.shape
    black_id = 8
    threshold = 0.4

    for x in range(width):
        color_counts = [np.count_nonzero(terrain_grid[x, :] == black_id)]

        total_pixels = height
        color_percentages = [count / total_pixels for count in color_counts]

        # Replace horizon lines
        if any(percentage >= threshold for percentage in color_percentages):
            terrain_grid[x, terrain_grid[x, :] == black_id] = 6

    print(f"Horizon line removal: {round((time.time() - seconds), 3)} s")
    return terrain_grid


# -=-=-=-=-=-=-=-
# Step 5: Calculating terrain costs

def calculate_terrain_costs(terrain_grid, relief_coords, color_costs, main_colors):
    seconds = time.time()
    width, height = terrain_grid.shape
    terrain_costs = np.zeros((width, height), dtype=float)

    # Layer 1 - raw terrain grid
    for i in range(width):
        for j in range(height):
            color_name = main_colors[terrain_grid[i, j]]
            terrain_costs[i, j] = color_costs[color_name]
    print(f"Terrain grid to terrain costs: {round((time.time() - seconds), 3)} s")

    # Layer 2 - height map (terrain, relief)
    seconds2 = time.time()
    for coordinate in relief_coords:
        i, j = coordinate
        if terrain_costs[i, j] != 10:
            terrain_costs[i, j] += 8

    print(f"Artificial LIDAR processing: {round((time.time() - seconds2), 3)} s")
    return terrain_costs


def create_graph_from_terrain(terrain_costs):
    rows, cols = terrain_costs.shape
    G = nx.DiGraph()

    for row in range(rows):
        for col in range(cols):
            node = (row, col)
            # Define neighbors: up, down, left, right
            neighbors = [
                (row + 1, col), (row - 1, col), 
                (row, col + 1), (row, col - 1)
            ]
            # Add diagonal neighbors if they are valid
            for nr, nc in neighbors:
                if 0 <= nr < rows and 0 <= nc < cols:  # Check bounds
                    weight = terrain_costs[nr, nc]
                    G.add_edge(node, (nr, nc), weight=weight)
            # Define diagonal neighbors
            diagonal_neighbors = [
                (row + 1, col + 1), (row - 1, col - 1),
                (row + 1, col - 1), (row - 1, col + 1)
            ]
            # Add valid diagonal neighbors
            for nr, nc in diagonal_neighbors:
                if 0 <= nr < rows and 0 <= nc < cols:  # Check bounds
                    # Check if both adjacent cells are traversable
                    if terrain_costs[nr, col] > 0 and terrain_costs[row, nc] > 0:
                        weight = terrain_costs[nr, nc]
                        G.add_edge(node, (nr, nc), weight=weight)
    return G

def find_path(terrain_costs, start_point, end_point):
    G = create_graph_from_terrain(terrain_costs)
    path = nx.astar_path(G, start_point, end_point, weight='weight')
    cost = nx.astar_path_length(G, start_point, end_point, weight='weight')
    return path, cost

def calculate_path(terrain_costs, start_point, end_point):
    seconds = time.time()
    path, cost = find_path(terrain_costs, start_point, end_point)
    print(f"Calculating A* path: {round((time.time() - seconds), 3)} s")
    return path[::-1], cost

def main():
    # Selecting a map file
    file_path = filedialog.askopenfilename(title="Select Map Image", filetypes=[("PNG files", "*.png")])
    if file_path:
        map_image = Image.open(file_path)
        
        root = tk.Tk()
        app = PointSelectionApp(root, map_image)
        root.mainloop()

        # Get the selected points
        raw_start_point = (int(app.points[0][0]), int(app.points[0][1]))
        raw_end_point = (int(app.points[1][0]), int(app.points[1][1]))
        cropped_map_image, start_point, end_point = crop_map_around_points(map_image, raw_start_point, raw_end_point)

        # Image processing
        raw_terrain_grid = process_image(cropped_map_image)
        raw_terrain_grid = coonnect_trails(raw_terrain_grid)
        relief_coords = artificial_lidar(raw_terrain_grid)
        terrain_grid = horizon_lines(raw_terrain_grid)

        # Calculate the combined terrain costs
        terrain_costs = calculate_terrain_costs(terrain_grid, relief_coords, color_costs, main_colors)

        # Calculate the lowest cost path
        path, cost = calculate_path(terrain_costs, start_point, end_point)

        # Plot the cropped map with the lowest cost path
        plt.figure()
        plt.imshow(cropped_map_image)
        plt.plot(*zip(*path), color='red', linewidth=2, linestyle='dotted', dash_capstyle='round', dashes=(2,))  # Plot the path
        plt.scatter(start_point[0], start_point[1], edgecolors='red', linewidths=2, s=100, marker='o') 
        plt.scatter(end_point[0], end_point[1], edgecolors='blue', linewidths=2, s=100, marker='o')  
        plt.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], color = "magenta", linewidth=1)  # Plot the connection
        plt.title(f'Optimal Route Choice with Cost: {round(cost,2)}')
        plt.show()
        print("Done!")

if __name__ == "__main__":
    main()
