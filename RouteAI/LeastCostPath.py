from PIL import Image, ImageTk
import numpy as np
import heapq
import tkinter as tk
from tkinter import filedialog
from tkinter.simpledialog import Dialog
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import time

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
    "yellow": 1.2,
    "orange": 1.4,
    "light_green": 3,
    "green": 4.5,
    "dark_green": 6,
    "light_blue": 5,
    "blue": 5,
    "olive": 100,
    "black": 1.3,
    "brown1": 20,
    "brown2": 20,
    "brown3": 20,
    "purple1": 20,
    "purple2": 20,
    "pink1": 20,
    "pink2": 20,
    "red": 20
}

class ModeSelectionDialog(Dialog):
    def __init__(self, master):
        self.map_type = None
        self.contours = None
        super().__init__(master)

    def body(self, master):
        self.title("Map type")
        self.geometry("300x200")

        # Calculate position for center of screen
        window_width = self.winfo_reqwidth()
        window_height = self.winfo_reqheight()
        position_right = int(self.winfo_screenwidth()/2 - window_width/2)
        position_down = int(self.winfo_screenheight()/2 - window_height/2)
        self.geometry("+{}+{}".format(position_right, position_down))

        tk.Label(master, text="Choose orienteering map type").pack()
        return None  

    def buttonbox(self):
        box = tk.Frame(self)

        forest_button = tk.Button(box, text="Forest", command=lambda: self.set_map_type("Forest"))
        sprint_button = tk.Button(box, text="Sprint", command=lambda: self.set_map_type("Sprint"))
        self.map_type_buttons = [forest_button, sprint_button]
        forest_button.pack(side=tk.LEFT, padx=5, pady=5)
        sprint_button.pack(side=tk.LEFT, padx=5, pady=5)

        contours_box = tk.Frame(self)
        m25_button = tk.Button(contours_box, text="2.5m", command=lambda: self.set_contours("2.5m"))
        m5_button = tk.Button(contours_box, text="5m", command=lambda: self.set_contours("5m"))
        self.contours_buttons = [m25_button, m5_button]
        m25_button.pack(side=tk.LEFT, padx=5, pady=5)
        m5_button.pack(side=tk.LEFT, padx=5, pady=5)

        continue_box = tk.Frame(self)
        tk.Button(continue_box, text="Continue", command=self.done).pack(side=tk.LEFT, padx=5, pady=5)

        box.pack()
        contours_box.pack()
        continue_box.pack()

    def set_map_type(self, map_type):
        self.map_type = map_type
        for button in self.map_type_buttons:
            if button.cget("text") == map_type:
                button.config(bg="green")
            else:
                button.config(bg="SystemButtonFace")

    def set_contours(self, contours):
        self.contours = contours
        for button in self.contours_buttons:
            if button.cget("text") == contours:
                button.config(bg="green")
            else:
                button.config(bg="SystemButtonFace")

    def done(self):
        self.result = (self.map_type, self.contours)
        self.master.destroy()


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
            self.canvas.coords(self.text_item, event.x +120, event.y + 20)


    def draw_polygon(self, event):
        x, y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        self.area.append((x, y))

        if 'polygon_item' in self.__dict__:
            self.canvas.delete(self.polygon_item)

        self.polygon_item = self.canvas.create_polygon(self.area, outline="black", fill='', width=3)

    def finish_polygon(self, event):

        if len(self.area) % 2 == 0:
            self.area.append(self.area[0])

        self.canvas.delete(self.polygon_item)
        self.polygon_item = self.canvas.create_polygon(self.area, outline="black", fill='', width=4)
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
def process_image(cropped_image, main_colors, main_color_values, color_tree, map_type):
    
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
            closest_color = find_closest_color(pixel, main_color_values, color_tree)
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
    return terrain_grid



def find_closest_color(pixel, main_color_values, color_tree):
    # Query the k-d tree to find the closest color
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



def main():
    # Select a map file
    file_path = filedialog.askopenfilename(title="Select Map Image", filetypes=[("PNG files", "*.png")])
    if file_path:
        map_image = Image.open(file_path)
        root = tk.Tk()

        # Choosing forest or sprint map
        dialog = ModeSelectionDialog(root)

        map_type = dialog.map_type
        contours = dialog.contours

        if map_type == 'Forest':
            pass

        elif map_type == 'Sprint':
            main_colors[18] = "gray"
            main_colors[19] = "road_orange"

            main_color_values["gray"] = (138, 138, 138)
            main_color_values["road_orange"] = (225, 195, 165)

            del color_costs["black"]
            color_costs["black"] = 100 # Impassable
            color_costs["road_orange"] = 1.1
            color_costs["gray"] = 100 # Impassable

        if contours == '2.5m':
            keys = ["brown1", "brown2", "brown3", "pink1", "pink2", "purple1", "purple2", "red"]

            for key in keys:
                color_costs[key] = 10

        elif contours == '5m':
            pass
        

        # Create the GUI for selecting the start and end points
        root = tk.Tk()
        app = PointSelectionApp(root, map_image)
        root.mainloop()

        raw_start_point = (int(app.points[0][0]), int(app.points[0][1]))
        raw_end_point = (int(app.points[1][0]), int(app.points[1][1]))

        final_time = time.time()

        # Crop the map around the points
        cropped_image, start_point, end_point = crop_map_around_points(map_image, raw_start_point, raw_end_point, app.area)


        # -=-=-=- Process image -=-=-=-=-

        color_tree = KDTree(list(main_color_values.values()))   # Create a k-d tree for the main colors to match the closest

        terrain_colors = process_image(cropped_image, main_colors, main_color_values, color_tree, map_type)
        terrain_costs = calculate_terrain_costs(terrain_colors, color_costs, main_colors)

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
