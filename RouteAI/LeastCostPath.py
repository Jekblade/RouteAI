from PIL import Image, ImageTk
import numpy as np
import heapq
import tkinter as tk
from tkinter import filedialog
from tkinter.simpledialog import Dialog
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import time
import math

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
    "yellow": 1.3,
    "orange": 1.4,
    "light_green": 3,
    "green": 4.5,
    "dark_green": 6,
    "light_blue": 5,
    "blue": 5,
    "olive": 5,
    "black": 1,
    "brown1": 20,
    "brown2": 20,
    "brown3": 20,
    "purple1": 20,
    "purple2": 20,
    "pink1": 20,
    "pink2": 20,
    "red": 20
}



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




# WHITE BALANCE CALIBRATION
def load_image(image_path):
    image = Image.open(image_path)
    image_np = np.array(image)
    return image_np

def apply_white_balance(image_np, white_pixel):
    if len(white_pixel) == 4:
        white_pixel = white_pixel[:3]  # ingore alfa

    ideal_white = np.array([255, 255, 255])

    # If the white pixel is close to white, return the original image
    if np.allclose(white_pixel, ideal_white, atol=10): 
        return image_np

    # Calculate white balance correction factor
    error = ideal_white / white_pixel
    white_balanced_image = np.clip(image_np[:, :, :3] * error, 0, 255).astype(np.uint8)  # Apply to RGB values only
    return white_balanced_image



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


# IMAGE PROCESSING
def process_image(cropped_image, main_colors, main_color_values, color_tree):
    
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
            closest_color = find_closest_color(pixel, main_color_values)
            terrain_grid[x, y] = list(main_colors.keys())[list(main_colors.values()).index(closest_color)]
    
    print(f"    Processing image ({width}x{height} pixels): {round((time.time() - process_time), 3)}s", end='\r')

    # Step 2: Check for horizon lines and remove them
    black_id = 9
    blue_id = 7
    light_blue_id = 6
    threshold = 0.5
    width, height = terrain_grid.shape
    columns_to_delete = []

    for y in range(height):
        black_pixel_count = np.count_nonzero(terrain_grid[:, y] == black_id)
        blue_pixel_count = np.count_nonzero(terrain_grid[:, y] == blue_id)
        light_blue_pixel_count = np.count_nonzero(terrain_grid[:, y] == light_blue_id)
        total_pixels = width
        black_pixel_percentage = black_pixel_count / total_pixels
        blue_pixel_percentage = blue_pixel_count / total_pixels
        light_blue_pixel_percentage = light_blue_pixel_count / total_pixels

        if black_pixel_percentage >= threshold or blue_pixel_percentage >= threshold or light_blue_pixel_percentage >= threshold:
            columns_to_delete.append(y)

    terrain_grid = np.delete(terrain_grid, y, axis=1)
    width, height = terrain_grid.shape


    # Step 3: Expanding lakes and rivers to remove black borders
    lakes_time = time.time()
    light_blue_id = 6
    blue_id = 7
    black_id = 9
    yellow_id = 1
    orange_id = 2

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            if terrain_grid[x, y] in [light_blue_id, blue_id, yellow_id, orange_id]:
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < width and 0 <= ny < height and terrain_grid[nx, ny] != blue_id:
                            if terrain_grid[nx, ny] == black_id or terrain_grid[nx, ny] == yellow_id or terrain_grid[nx, ny] == orange_id:
                                terrain_grid[nx, ny] = blue_id

    print(f"\n    Removing lake and river borders: {round((time.time() - lakes_time), 3)}s")
    return terrain_grid


# Find the closest color using KDTree
def find_closest_color(pixel, main_color_values):

    color_tree = KDTree(list(main_color_values.values()))

    _, index = color_tree.query(pixel[:3])
    closest_color = list(main_color_values.keys())[index]
    return closest_color



# Calculating the terrain costs
def calculate_terrain_costs(terrain_colors, color_costs, main_colors):
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


import tkinter as tk
from PIL import Image, ImageTk

class RouteAI:
    def __init__(self, master, map_image):

        self.master = master
        self.master.title("Selection of start point and end point")

        screen_width = self.master.winfo_screenwidth()
        screen_height = self.master.winfo_screenheight()

        max_width = int(screen_width * 0.6)  # 60% of the screen width
        max_height = int(screen_height * 0.9)  # Full screen height

        # Resize the image
        img_width, img_height = map_image.size
        if img_height > max_height or img_width > max_width:
            # keeping aspect ratio
            ratio = min(max_width / img_width, max_height / img_height)
            new_width = int(img_width * ratio)
            new_height = int(img_height * ratio)
            map_image = map_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        self.canvas = tk.Canvas(master)
        self.canvas.pack(side="left", fill="y")

        self.map_image = map_image
        self.map_photo = ImageTk.PhotoImage(map_image)
        self.image_item = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.map_photo)

        canvas_width = self.map_photo.width()
        canvas_height = self.map_photo.height()
        self.canvas.config(width=canvas_width, height=canvas_height)

        self.map_type = tk.StringVar()
        self.map_type.set("Forest")  # Default selection
        self.contours = tk.StringVar()
        self.contours.set("2.5m")  # Default selection


        button_frame = tk.Frame(master)
        button_frame.pack(side=tk.RIGHT, padx=10, pady=10)

        tk.Label(button_frame, text="Map Type:").grid(row=0, column=0, columnspan=2)
        tk.Radiobutton(button_frame, text="Forest", variable=self.map_type, value="Forest").grid(row=1, column=0)
        tk.Radiobutton(button_frame, text="Sprint", variable=self.map_type, value="Sprint").grid(row=1, column=1)

        tk.Label(button_frame, text="").grid(row=2, column=0)

        tk.Label(button_frame, text="Contours:").grid(row=3, column=0, columnspan=2)
        tk.Radiobutton(button_frame, text="2.5m", variable=self.contours, value="2.5m").grid(row=4, column=0)
        tk.Radiobutton(button_frame, text="5m", variable=self.contours, value="5m").grid(row=4, column=1)

        for i in range(5, 6):
            tk.Label(button_frame, text="").grid(row=i, column=0)

        self.another_route_button = tk.Button(button_frame, text="Create a new route", command=self.another_route)
        self.another_route_button.grid(row=7, column=0, columnspan=2)

        # Add StringVars for the progress, optimal route cost, and approximated time
        self.progress = tk.StringVar()
        self.progress.set("Progress: 0%")
        self.optimal_route_cost = tk.StringVar()
        self.optimal_route_cost.set("Optimal route cost: N/A")
        self.approximated_time = tk.StringVar()
        self.approximated_time.set("Approximated time: N/A")

        # Add labels for the progress, optimal route cost, and approximated time
        tk.Label(button_frame, textvariable=self.progress).grid(row=8, column=0, columnspan=2)
        tk.Label(button_frame, textvariable=self.optimal_route_cost).grid(row=9, column=0, columnspan=2)
        tk.Label(button_frame, textvariable=self.approximated_time).grid(row=10, column=0, columnspan=2)


        self.points = []
        self.area = []
        self.start_point_selected = False
        self.finish_point_selected = False
        self.drawing_polygon = False
        self.canvas.bind("<Button-1>", self.add_point)


    def another_route(self):
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.map_photo)

        self.points = []
        self.area = []
        self.start_point_selected = False
        self.finish_point_selected = False
        self.drawing_polygon = False

        self.canvas.bind("<Button-1>", self.add_point)


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
                self.select_area(event)

    def select_area(self, event):
        if self.finish_point_selected:
            self.text_item = self.canvas.create_text(event.x, event.y, text="Select an analysis area", fill="black", font=("Arial", 20))

            # Delaying the draw_polygon() to prevent the start point from being added twice
            self.master.after(100, lambda: self.canvas.bind("<B1-Motion>", self.draw_polygon))
            self.canvas.bind("<ButtonRelease-1>", self.finish_polygon)
            self.canvas.bind("<Motion>", self.move_text)
            self.canvas.bind("<Button-1>", self.remove_text)

    def remove_text(self, event):
        self.canvas.delete(self.text_item)

    def move_text(self, event):
        if self.finish_point_selected:
            self.canvas.coords(self.text_item, event.x + 120, event.y + 20)

    def draw_polygon(self, event):
        self.drawing_polygon = True
        x, y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        self.area.append((x, y))

        if 'polygon_item' in self.__dict__:
            self.canvas.delete(self.polygon_item)

        self.polygon_item = self.canvas.create_polygon(self.area, outline="black", fill='', width=3)


    def finish_polygon(self, event):
        if not self.drawing_polygon:
            return
        
        self.canvas.delete(self.polygon_item)
        self.polygon_item = self.canvas.create_polygon(self.area, outline="black", fill='', width=3)
        self.canvas.unbind("<B1-Motion>")
        self.canvas.unbind("<ButtonRelease-1>")
        self.canvas.unbind("<Motion>")
        self.canvas.unbind("<Button-1>")

        self.complete_selection(self.area)


    def complete_selection(self, area):
        final_time = time.time()
        raw_start_point = (int(self.points[0][0]), int(self.points[0][1]))
        raw_end_point = (int(self.points[1][0]), int(self.points[1][1]))

        if self.map_type.get() == 'Sprint':
            main_colors[18] = "gray"
            main_colors[19] = "road_orange"
            main_colors[20] = "passable_gray"
            main_colors[21] = "pale_road"
            main_color_values["gray"] = (138, 138, 138)
            main_color_values["road_orange"] = (225, 195, 165)
            main_color_values["passable_gray"] = (210, 210, 210)
            main_color_values["pale_road"] = (250, 248, 224)

            color_costs.update({
            "black": 100,  # Impassable
            "road_orange": 1.1,
            "gray": 100,  # Impassable
            "passable_gray": 1.1,
            "pale_road": 1.1
            })

            keys = ["purple1", "purple2", "pink1", "pink2", "red"]
            for key in keys:
                color_costs[key] = 100

        if self.map_type.get() == 'Forest':
            main_colors.pop("gray", None)
            main_colors.pop("road_orange", None)
            main_colors.pop("passable_gray", None)
            main_colors.pop("pale_road", None)

            main_color_values.pop("gray", None)
            main_color_values.pop("road_orange", None)
            main_color_values.pop("passable_gray", None)
            main_color_values.pop("pale_road", None)

            color_costs.pop("gray", None)
            color_costs.pop("road_orange", None)
            color_costs.pop("passable_gray", None)
            color_costs.pop("pale_road", None)
            color_costs["black"] = 1.3

            keys = ["purple1", "purple2", "pink1", "pink2", "red"]
            for key in keys:
                color_costs[key] = 20

        if self.contours.get() == '2.5m' or self.contours.get() == '5m':

            keys = ["brown1", "brown2", "brown3"]
            for key in keys:
                color_costs[key] = 10 if self.contours.get() == '2.5m' else 20
            
            keys = ["purple1", "purple2", "pink1", "pink2", "red"]
            for key in keys:
                color_costs[key] = 20


        cropped_image, start_point, end_point = crop_map_around_points(self.map_image, raw_start_point, raw_end_point, area)

        color_tree = KDTree(list(main_color_values.values()))   # Create a k-d tree for the main colors to match the closest

        terrain_colors = process_image(cropped_image, main_colors, main_color_values, color_tree)

        # Calculate terrain costs
        terrain_costs = calculate_terrain_costs(terrain_colors, color_costs, main_colors)

        path, cost = calculate_lowest_cost_path(terrain_costs, start_point, end_point)

        print(f"                    *\n    DONE! Total time: {round((time.time() - final_time), 3)}s")

        # Convert the cropped image to a PhotoImage and display it on the canvas
        self.cropped_photo = ImageTk.PhotoImage(cropped_image)
        self.image_item = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.cropped_photo)

        # Add a border around the cropped photo
        border_width = 5
        border_color = "#FF8C00"  # more darker orange
        x1, y1, x2, y2 = self.canvas.bbox(self.image_item)
        x1 -= border_width
        y1 -= border_width
        x2 += border_width
        y2 += border_width
        self.canvas.create_rectangle(x1, y1, x2, y2, outline=border_color, width=border_width)

        # Draw the optimal route on the canvas
        for i in range(len(path) - 1):
            self.canvas.create_line(path[i][0], path[i][1], path[i+1][0], path[i+1][1], fill='red', width=3)

        start_point_item = self.canvas.create_oval(start_point[0] - 10, start_point[1] - 10, start_point[0] + 10, start_point[1] + 10, outline="red", width=2, fill='')
        end_point_item = self.canvas.create_oval(end_point[0] - 10, end_point[1] - 10, end_point[0] + 10, end_point[1] + 10, outline="blue", width=2, fill='')

        # Calculate the direction of the line
        dx = end_point[0] - start_point[0]
        dy = end_point[1] - start_point[1]
        length = math.sqrt(dx**2 + dy**2)
        dx /= length
        dy /= length

        # Adjust the start and end points
        start_point = (start_point[0] + 10 * dx, start_point[1] + 10 * dy)
        end_point = (end_point[0] - 10 * dx, end_point[1] - 10 * dy)

        # Draw the connection between the adjusted start point and end point
        connection_item = self.canvas.create_line(start_point[0], start_point[1], end_point[0], end_point[1], fill='green', width=3)

        self.progress.set("Progress: 100%")
        self.optimal_route_cost.set(f"Optimal route cost: {round(cost, 2)}")
        self.approximated_time.set(f"Approximated time: {round(cost / 180)}min")



def main():
    # Select a map file
    file_path = filedialog.askopenfilename(title="Select Map Image", filetypes=[("PNG files", "*.png")])
    if file_path:
        map_image = Image.open(file_path)

        

        root = tk.Tk()
        app = RouteAI(root, map_image)
        root.mainloop()

if __name__ == "__main__":
    main()
