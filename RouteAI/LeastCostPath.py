from PIL import Image, ImageTk
import numpy as np
import heapq
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import time

# Define main colors for categories
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
    10: "brown1",
    11: "brown2",
    12: "brown3",
    13: "purple1",
    14: "purple2",
    15: "pink1",
    16: "pink2",
    17: "red"
}

# RGB values for main colors
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
    "brown1": (190, 105, 40),
    "brown2": (180, 172, 112),
    "brown3": (130, 140, 75),
    "purple1": (136, 0, 160),
    "purple2": (150, 70, 140),
    "pink1": (215, 0, 120),
    "pink2": (195, 31, 255),
    "red": (135, 10, 50)
}

# Costs associated with terrain runnability
color_costs = {
    "white": 1.6, 
    "yellow": 1.2,
    "orange": 1.4,
    "light_green": 3,
    "green": 4.5,
    "dark_green": 6,
    "light_blue": 5,
    "blue": 5,
    "olive": 100,
    #------------------------
    "black": 1,
    "brown1": 30,
    "brown2": 30,
    "brown3": 30,
    "purple1": 50,
    "purple2": 50,
    "pink1": 50,
    "pink2": 50,
    "red": 50
}



class PointSelectionApp:
    def __init__(self, master, map_image):
        self.master = master
        self.master.title("Selection of start point and end point")

        self.points = []
        self.area = []
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
        self.master.bind("<Return>", self.select_area)


    def add_point(self, event):
        x, y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)

        if len(self.points) <= 2:
           
            if not self.start_point_selected:

                # Draw the start point
                start_point_item = self.canvas.create_oval(x - 10, y - 10, x + 10, y + 10, outline="red", width=2, fill='')
                self.points.append((x, y, start_point_item))
                self.start_point_selected = True

            # Do the same for the finish point
            elif self.start_point_selected and not self.finish_point_selected:

                finish_point_item = self.canvas.create_oval(x - 10, y - 10, x + 10, y + 10, outline="blue", width=2, fill='')
                self.points.append((x, y, finish_point_item))
                self.finish_point_selected = True

    def select_area(self, event):
        if self.finish_point_selected:
            # Delaying the draw_polygon() to prevent the start point from being added twice
            self.master.after(100, lambda: self.canvas.bind("<B1-Motion>", self.draw_polygon))
            self.canvas.bind("<ButtonRelease-1>", self.finish_polygon)

    def draw_polygon(self, event):
        x, y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        self.area.append((x, y))

        if 'polygon_item' in self.__dict__:
            self.canvas.delete(self.polygon_item)

        self.polygon_item = self.canvas.create_polygon(self.area, outline="purple", fill='', width=4)

    def finish_polygon(self, event):

        if len(self.area) % 2 == 0:
            self.area.append(self.area[0])

        self.canvas.delete(self.polygon_item)
        self.polygon_item = self.canvas.create_polygon(self.area, outline="purple", fill='', width=4)
        self.canvas.unbind("<B1-Motion>")
        self.canvas.unbind("<ButtonRelease-1>")
        self.master.bind("<Return>", self.complete_selection)

    def complete_selection(self, event):
        self.canvas.unbind("<B1-Motion>")
        self.master.unbind("<Return>")
        self.master.destroy()


def crop_map_around_points(map_image, raw_start_point, raw_end_point, area):
    # Extract coordinates
    start_x, start_y = raw_start_point
    end_x, end_y = raw_end_point

    map_width, map_height = map_image.size

    # Taking min max coords of the selected area
    x_coords, y_coords = zip(*area)

    min_x_selected = int(min(x_coords))
    max_x_selected = int(max(x_coords))
    min_y_selected = int(min(y_coords))
    max_y_selected = int(max(y_coords))


    # Determine the bounding box for cropping
    min_x = max(0, min_x_selected)
    max_x = min(map_width, max_x_selected)
    min_y = max(0, min_y_selected)
    max_y = min(map_height,max_y_selected)

    cropped_image = map_image.crop((min_x, min_y, max_x, max_y))

    # Recalibrating start and end points
    start_point = (start_x - min_x, start_y - min_y)
    end_point = (end_x - min_x, end_y - min_y)

    return cropped_image, start_point, end_point


# IMAGE PROCESSING
def process_image(cropped_image):
    
    process_time = time.time()
    progress = 0
    map_image_np = np.array(cropped_image)
    height, width, _ = map_image_np.shape
    terrain_grid = np.zeros((width, height), dtype=np.uint8)

    # Step 1: Process all pixels
    print("\n\n\n\n      -==- FINDING AN OPTIMAL ROUTE -==-\n")

    for y in range(height):
        progress = int((y / height) * 100)
        print(f"    Processing image ({width}x{height} pixels): {round(progress,1)}%", end='\r')
    
        for x in range(width):
            pixel = map_image_np[y, x]
            closest_color = find_closest_color(pixel)
            terrain_grid[x, y] = list(main_colors.keys())[list(main_colors.values()).index(closest_color)]
    
    print(f"    Processing image ({width}x{height} pixels): {round((time.time() - process_time), 3)}s", end='\r')

    # Step 2: Check for black horizon lines and change them to brown
    black_id = 9
    threshold = 0.3
    width, height = terrain_grid.shape

    for y in range(height):
        black_pixel_count = np.count_nonzero(terrain_grid[:, y] == black_id)
        total_pixels = width
        black_pixel_percentage = black_pixel_count / total_pixels

        if black_pixel_percentage >= threshold:
            terrain_grid[:, y][terrain_grid[:, y] == black_id] = 11  # Change black to brown


    # Step 3: Expanding lakes and rivers to remove black borders
    lakes_time = time.time()
    light_blue_id = 6
    blue_id = 7
    black_id = 9

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            if terrain_grid[x, y] in [light_blue_id, blue_id]:
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < width and 0 <= ny < height and terrain_grid[nx, ny] != blue_id and terrain_grid[nx, ny] == black_id:
                            terrain_grid[nx, ny] = blue_id
    print(f"\n    Removing lake and river borders: {round((time.time() - lakes_time), 3)}s")

    # Step 4: Black pixel classification - connecting roads and trails

    terrain_grid_final = np.copy(terrain_grid)
    trails_time = time.time()

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

    print(f"    Connecting trails and paths: {round((time.time() - trails_time), 3)}s")
    return terrain_grid_final


# Create a k-d tree for the main colors
color_tree = KDTree(list(main_color_values.values()))

def find_closest_color(pixel):
    # Query the k-d tree to find the closest color
    _, index = color_tree.query(pixel[:3])

    closest_color = list(main_color_values.keys())[index]
    return closest_color



# Calculating the terrain costs
def calculate_terrain_costs(terrain_colors):
    width, height = terrain_colors.shape
    terrain_costs = np.zeros((width, height), dtype=float)
    terrain_time = time.time()

    for i in range(width):
        for j in range(height):
            color_name = main_colors[terrain_colors[i, j]]
            terrain_costs[i, j] = color_costs[color_name]

    print(f"                    *\n    Map converted into cost array: {round((time.time() - terrain_time), 3)}s")
    return terrain_costs



# Finding the optimal route
def calculate_lowest_cost_path(terrain_costs, start_point, end_point):
    route_time = time.time()

    width, height = terrain_costs.shape[0], terrain_costs.shape[1]
    dp = np.full((width, height), np.inf)
    dp[start_point] = terrain_costs[start_point]

    state = np.zeros((width, height))

    queue = [(terrain_costs[start_point], start_point)]
    while queue:
        cost, (x, y) = heapq.heappop(queue)
        if (x, y) == end_point:
            state[x, y] = 2
            break  # Path found, no need to continue
        else:
            state[x, y] = 1
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height:
                if dx != 0 and dy != 0:  # Diagonal movement buffed due to Pythagorean theorem
                    next_cost = cost + terrain_costs[nx, ny] * 1.41
                else:
                    next_cost = cost + terrain_costs[nx, ny]
                if next_cost < dp[nx, ny]:
                    dp[nx, ny] = next_cost
                    heapq.heappush(queue, (next_cost, (nx, ny)))

    path = []
    x, y = end_point
    while (x, y) != start_point:
        path.append((x, y))
        min_cost = np.inf
        nx, ny = -1, -1
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            nx_, ny_ = x + dx, y + dy
            if 0 <= nx_ < width and 0 <= ny_ < height and dp[nx_, ny_] < min_cost:
                min_cost = dp[nx_, ny_]
                nx, ny = nx_, ny_
        x, y = nx, ny
    path.append(start_point)

    cost = dp[end_point]

    print(f"    Finding the optimal route: {round((time.time() - route_time), 3)} s")

    return path[::-1], cost



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

        final_time = time.time()
        # Crop the map around the points
        cropped_image, start_point, end_point = crop_map_around_points(map_image, raw_start_point, raw_end_point, app.area)

        # Process image
        terrain_colors = process_image(cropped_image)
        terrain_costs = calculate_terrain_costs(terrain_colors)

        # Calculate the lowest cost path
        path, cost = calculate_lowest_cost_path(terrain_costs, start_point, end_point)

        print(f"                    *\n    DONE! Total time: {round((time.time() - final_time), 3)}s")

        # Plot the cropped map with the lowest cost path
        plt.figure()
        plt.imshow(cropped_image)
        plt.plot(*zip(*path), color='red', linewidth=3, linestyle='dotted', dashes=(2,))  # Plot the path
        plt.scatter(start_point[0], start_point[1], edgecolors='red', linewidths=2, s=100, marker='o') 
        plt.scatter(end_point[0], end_point[1], edgecolors='blue', linewidths=2, s=100, marker='o')  
        plt.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], color = "magenta", linewidth=1)  # Plot the connection
        plt.title(f'Least Cost path: {round(cost,2)} ||| Approximated time: {round(cost/180)}min')
        plt.show()

if __name__ == "__main__":
    main()
