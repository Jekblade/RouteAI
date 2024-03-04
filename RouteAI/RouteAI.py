from PIL import Image, ImageTk
import numpy as np
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
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

# Costs associated with terrain runnability
color_costs = {
    "white": 1,
    "light_green": 2,
    "green": 4,
    "dark_green": 8,
    "orange": 0.5,
    "black": 0.01,
    "yellow": 0.3,
    "blue": 20,
    "brown": 40,
    "dark_brown": 40,
    "purple": 30,
    "olive": 0,
    "pink": 30
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
            terrain_grid[x, :] = 2

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

    return terrain_grid_final


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

    # Get the size of the map image
    map_width, map_height = map_image.size

    # Determine the bounding box for cropping
    min_x = max(0, min(start_x, end_x) - buffer_size)
    max_x = min(map_width, max(start_x, end_x) + buffer_size)
    min_y = max(0, min(start_y, end_y) - buffer_size)
    max_y = min(map_height, max(start_y, end_y) + buffer_size)

    # Crop the map image
    cropped_image = map_image.crop((min_x, min_y, max_x, max_y))


    # Update the adjusted start and end points
    start_point = (start_x - min_x, start_y - min_y)
    end_point = (end_x - min_x, end_y - min_y)

    # Calculate crop offset
    crop_offset = (min_x, min_y)

    return cropped_image, start_point, end_point, crop_offset


# Calculating the terrain costs
def calculate_terrain_costs(terrain_colors):
    width, height = terrain_colors.shape
    terrain_costs = np.zeros((width, height), dtype=float)

    for i in range(width):
        for j in range(height):
            color_name = main_colors[terrain_colors[i, j]]
            terrain_costs[i, j] = color_costs[color_name]

    return terrain_costs

def calculate_path(terrain_costs, start_point, end_point):
    width, height = terrain_costs.shape[0], terrain_costs.shape[1]
    dp = np.full((width, height), np.inf)
    dp[start_point] = terrain_costs[start_point]

    # Determine the direction of the end point relative to the start point
    dx = 1 if end_point[0] > start_point[0] else -1
    dy = 1 if end_point[1] > start_point[1] else -1

    # Fill the dynamic programming array based on the direction of the end point
    for i in range(start_point[0] + dx, end_point[0] + dx, dx):
        dp[i, start_point[1]] = dp[i - dx, start_point[1]] + terrain_costs[i, start_point[1]]
    for j in range(start_point[1] + dy, end_point[1] + dy, dy):
        dp[start_point[0], j] = dp[start_point[0], j - dy] + terrain_costs[start_point[0], j]

    for i in range(start_point[0] + dx, end_point[0] + dx, dx):
        for j in range(start_point[1] + dy, end_point[1] + dy, dy):
            # Avoid olive green areas by setting their cost to infinity
            if terrain_costs[i, j] == color_costs["olive"]:
                dp[i, j] = np.inf
            else:
                # Consider all directions except going backward
                candidates = [dp[i - dx, j], dp[i, j - dy], dp[i - dx, j - dy]]
                if dx == 1:
                    candidates.append(dp[i - dx, j - dy])
                elif dx == -1:
                    candidates.append(dp[i, j - dy])
                dp[i, j] = min(candidates) + terrain_costs[i, j]

    # Backtrack to find the lowest cost path
    path = [(end_point[0], end_point[1])]  # Start from the end point
    i, j = end_point[0], end_point[1]
    while i != start_point[0] or j != start_point[1]:
        candidates = [(i - dx, j), (i, j - dy), (i - dx, j - dy)]
        if dx == 1:
            candidates.append((i - dx, j - dy))
        elif dx == -1:
            candidates.append((i, j - dy))
        # Filter out candidates that are out of bounds or have infinite cost
        valid_candidates = [(x, y) for x, y in candidates if 0 <= x < width and 0 <= y < height and dp[x, y] != np.inf]
        # Find the minimum neighbor among the valid candidates
        if valid_candidates:
            min_neighbor = min((dp[x, y], x, y) for x, y in valid_candidates)
            i, j = min_neighbor[1:]
            path.append((i, j))
        else:
            # If no valid candidates are found, break out of the loop and search around the olive green area
            olive_green_indices = np.where(terrain_costs == color_costs["olive"])
            olive_green_points = list(zip(olive_green_indices[0], olive_green_indices[1]))
            for olive_i, olive_j in olive_green_points:
                path.append((olive_i, olive_j))
            break

    # Add the starting point, remove first and last 10
    path.append(start_point)
    path = path[6:-6]

    return path[::-1], dp[end_point]


def main():
    # Select a map file
    file_path = filedialog.askopenfilename(title="Select Map Image", filetypes=[("PNG files", "*.png")])
    if file_path:
        map_image = Image.open(file_path)
        
        root = tk.Tk()
        app = PointSelectionApp(root, map_image)
        root.mainloop()

        # Get the selected points
        raw_start_point = (int(app.points[0][0]), int(app.points[0][1]))
        raw_end_point = (int(app.points[1][0]), int(app.points[1][1]))

        # Crop the map around the points
        cropped_map_image, start_point, end_point, crop_offset = crop_map_around_points(map_image, raw_start_point, raw_end_point)
        terrain_colors = process_image(cropped_map_image)

        # Calculate the terrain costs
        terrain_costs = calculate_terrain_costs(terrain_colors)

        # Calculate the lowest cost path
        path, cost = calculate_path(terrain_costs, start_point, end_point)

        # Plot the cropped map with the lowest cost path
        plt.figure()
        plt.imshow(cropped_map_image)
        plt.plot(*zip(*path), color='red', linewidth=2)  # Plot the path
        plt.scatter(start_point[0], start_point[1], edgecolors='red', linewidths=2, s=100, marker='o') 
        plt.scatter(end_point[0], end_point[1], edgecolors='blue', linewidths=2, s=100, marker='o')  
        plt.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], color = "magenta", linewidth=1)  # Plot the connection
        plt.title('Optimal Route choice')
        plt.show()

if __name__ == "__main__":
    main()
