from PIL import Image, ImageTk
import numpy as np
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt

# Check for vertically stacked pixels and remove columns accordingly
black_id = 4
olive_id = 7
blue_id = 6
threshold = 0.6

colors_to_check = [black_id, olive_id, blue_id]

for x in range(width):
    # Count the number of pixels for each color in the current column
    color_counts = [np.count_nonzero(terrain_grid[x, :] == color_id) for color_id in colors_to_check]
    total_pixels = height

    color_percentages = [count / total_pixels for count in color_counts]

    if any(percentage >= threshold for percentage in color_percentages):
        terrain_grid[x, :] = 0


# Define main colors for categories
main_colors = {
    0: "white",
    1: "light_green",
    2: "green",
    3: "orange",
    4: "black",
    5: "yellow",
    6: "blue",
    7: "brown",
    8: "dark_brown",
    9: "purple",
    10: "olive"
}

# RGB values for main colors
main_color_values = {
    "white": (255, 255, 255),
    "light_green": (204, 224, 191),
    "green": (150, 200, 150),
    "orange": (245, 210, 150),
    "black": (50, 50, 50),
    "yellow": (250, 200, 90),
    "blue": (80, 180, 220),
    "brown": (210, 175, 150),
    "dark_brown": (170, 80, 30),
    "purple": (130, 20, 115),
    "olive": (157, 177, 59)
}

# Costs associated with terrain runnability
color_costs = {
    "white": 0.2,
    "light_green": 0.4,
    "green": 0.8,
    "orange": 0.1,
    "black": 0.01,
    "yellow": 0.1,
    "blue": 0.8,
    "brown": 40,
    "dark_brown": 40,
    "purple": 10,
    "olive": 0.2
}

def load_image(file_path):
    image = Image.open(file_path)
    return image

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

def process_image(map_image):
    map_image_np = np.array(map_image)
    height, width, _ = map_image_np.shape
    terrain_grid = np.zeros((width, height), dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            pixel = map_image_np[y, x]
            closest_color = find_closest_color(pixel)
            terrain_grid[x, y] = list(main_colors.keys())[list(main_colors.values()).index(closest_color)]
    
    # Check for vertically stacked pixels and remove columns accordingly
    black_id = 4
    olive_id = 7
    blue_id = 6
    threshold = 0.50

    colors_to_check = [black_id, olive_id, blue_id]
    columns_to_remove = []

    for x in range(width):
        # Count the number of pixels for each color in the current column
        color_counts = [np.count_nonzero(terrain_grid[x, :] == color_id) for color_id in colors_to_check]
        total_pixels = height

        color_percentages = [count / total_pixels for count in color_counts]

        if any(percentage >= threshold for percentage in color_percentages):
            columns_to_remove.append(x)

    terrain_grid = np.delete(terrain_grid, columns_to_remove, axis=0)

    return terrain_grid


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

        if len(self.points) <= 2 and 0 <= x <= self.canvas.winfo_width() and 0 <= y <= self.canvas.winfo_height():
           
            # If start point is not selected yet
            if not self.start_point_selected:
                # Remove previously selected start point if exists
                if len(self.points) > 0:
                    self.canvas.delete(self.points[-1][2])

                # Draw the points
                start_point_item = self.canvas.create_oval(x - 8, y - 8, x + 8, y + 8, outline="red", width=2, fill='')
                self.points.append((x, y, start_point_item))
                self.start_point_selected = True

            # Do the same for the finish point
            elif self.start_point_selected and not self.finish_point_selected:

                finish_point_item = self.canvas.create_oval(x - 8, y - 8, x + 8, y + 8, outline="blue", width=2, fill='')
                self.points.append((x, y, finish_point_item))
                self.finish_point_selected = True

    def complete_selection(self, event):
        if self.start_point_selected and self.finish_point_selected:
            self.master.quit()


def crop_map_around_points(map_image, start_point, end_point, buffer_size=100):
    # Extract coordinates
    start_x, start_y = start_point
    end_x, end_y = end_point

    # Get the size of the map image
    map_width, map_height = map_image.size

    # Determine the bounding box for cropping
    min_x = max(0, min(start_x, end_x) - buffer_size)
    max_x = min(map_width, max(start_x, end_x) + buffer_size)
    min_y = max(0, min(start_y, end_y) - buffer_size)
    max_y = min(map_height, max(start_y, end_y) + buffer_size)

    cropped_image = map_image.crop((min_x, min_y, max_x, max_y))



    # Update the adjusted start and end points
    adjusted_start_point = (start_x - min_x, start_y - min_y)
    adjusted_end_point = (end_x - min_x, end_y - min_y)

    # Calculate crop offset
    crop_offset = (min_x, min_y)

    return cropped_image, adjusted_start_point, adjusted_end_point, crop_offset


# Calculating the terrain costs
def calculate_terrain_costs(terrain_colors):
    width, height = terrain_colors.shape
    terrain_costs = np.zeros((width, height), dtype=float)

    for i in range(width):
        for j in range(height):
            color_name = main_colors[terrain_colors[i, j]]
            terrain_costs[i, j] = color_costs[color_name]

    return terrain_costs

def calculate_lowest_cost_path(terrain_costs, start_point, end_point):
    # Calculate the lowest cost path from start to end
    start_to_end_path, start_to_end_cost = _calculate_single_path(terrain_costs, start_point, end_point)

    # Calculate the lowest cost path from end to start
    end_to_start_path, end_to_start_cost = _calculate_single_path(terrain_costs, end_point, start_point)

    # Determine which path has the lowest cost
    if start_to_end_cost < end_to_start_cost:
        return start_to_end_path, start_to_end_cost
    else:
        return end_to_start_path, end_to_start_cost

def _calculate_single_path(terrain_costs, start_point, end_point):
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
        # Filter out candidates that are out of bounds
        valid_candidates = [(x, y) for x, y in candidates if 0 <= x < width and 0 <= y < height]
        # Find the minimum neighbor among the valid candidates
        if valid_candidates:
            min_neighbor = min((dp[x, y], x, y) for x, y in valid_candidates)
            i, j = min_neighbor[1:]
            path.append((i, j))

    # Add the starting point
    path.append(start_point)

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
        start_point = (int(app.points[0][0]), int(app.points[0][1]))
        end_point = (int(app.points[1][0]), int(app.points[1][1]))

        # Crop the map around the points
        cropped_map_image, adjusted_start_point, adjusted_end_point, crop_offset = crop_map_around_points(map_image, start_point, end_point)

        # Process the cropped map image
        terrain_colors = process_image(cropped_map_image)

        # Calculate the terrain costs
        terrain_costs = calculate_terrain_costs(terrain_colors)

        # Calculate the lowest cost path
        path, cost = calculate_lowest_cost_path(terrain_costs, adjusted_start_point, adjusted_end_point)

        # Plot the cropped map with the lowest cost path
        plt.figure()
        plt.imshow(cropped_map_image)
        plt.plot(*zip(*path), color='red', linewidth=2)  # Plot the path
        plt.scatter(adjusted_start_point[0], adjusted_start_point[1], edgecolors='red', linewidths=2, s=100, marker='o', alpha=0) 
        plt.scatter(adjusted_end_point[0], adjusted_end_point[1], edgecolors='blue', linewidths=2, s=100, marker='o', alpha=0)  
        plt.plot([adjusted_start_point[0], adjusted_end_point[0]], [adjusted_start_point[1], adjusted_end_point[1]], color = "magenta", linewidth=1)  # Plot the connection
        plt.title('Cropped Map with Lowest Cost Path')
        plt.show()

if __name__ == "__main__":
    main()
