import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import networkx as nx

class PointSelectionApp:
    def __init__(self, master, map_image):
        self.master = master
        self.master.title("Rogaining course planning with kNearestNeighbour")

        self.points = []
        self.red_point_selected = False

        self.canvas = tk.Canvas(master)
        self.canvas.pack(side="left", fill="y")

        # Scrollbars
        self.v_scrollbar = tk.Scrollbar(master, orient="vertical", command=self.canvas.yview)
        self.v_scrollbar.pack(side="right", fill="y")
        self.canvas.configure(yscrollcommand=self.v_scrollbar.set)

        self.map_photo = ImageTk.PhotoImage(map_image)
        self.image_item = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.map_photo)
        canvas_width = self.map_photo.width()
        canvas_height = self.map_photo.height()

        # Get the screen's height
        screen_height = master.winfo_screenheight()
        if canvas_height >= screen_height:
            canvas_height = master.winfo_screenheight()

        self.canvas.config(width=canvas_width, height=canvas_height)

        # Actions
        self.canvas.bind("<Button-1>", self.add_point)
        self.master.bind("<Return>", self.complete_selection)

        self.canvas.update_idletasks()
        self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))

        # Get the map width and height
        self.map_width = map_image.width
        self.map_height = map_image.height


    def add_point(self, event):
        x, y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        x += self.canvas.xview()[0]

        # Check if the click is within the bounds of the map canvas
        if 0 <= x <= self.map_width and 0 <= y <= self.map_height:
            # If the red point hasn't been selected yet, mark it as red
            if not self.red_point_selected:
                color = "red"
                self.red_point_selected = True
            else:
                color = "blue"

            # Add the point to the canvas
            point_item = self.canvas.create_oval(x - 10, y - 10, x + 10, y + 10, outline=color, width=2, fill='')

            # Store the coordinates of the point along with its canvas item
            self.points.append((x, y, point_item))

            
    def complete_selection(self, event):
        # Close the window
        self.master.destroy()

def draw_shortest_path(points, map_image):
    k = len(points) - 2
    red_point = np.array([points[0]])

    # Get the original points excluding the red point
    original_points = np.array(points[1:])

    # Initialize the nearest neighbor algorithm
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(original_points)

    # Start with the red point
    chain = [tuple(red_point[0])]

    visited = set()

    connections = []
    while len(visited) < len(original_points):
        # Find the nearest neighbors to the last point in the chain
        distances, indices = nbrs.kneighbors(np.array([chain[-1]]))

        # Find the nearest unvisited neighbor
        nearest_idx = None
        for idx in indices[0]:
            if tuple(original_points[idx]) not in visited:
                nearest_idx = idx
                break

        if nearest_idx is not None:
            # Add the nearest unvisited neighbor to the chain
            nearest_neighbor = original_points[nearest_idx]
            chain.append(tuple(nearest_neighbor))

            connections.append((chain[-2], chain[-1]))

            visited.add(tuple(original_points[nearest_idx]))


    # Plot the points and connections
    plt.figure(figsize=(8, 6))
    for connection in connections:
        plt.plot([connection[0][0], connection[1][0]], [connection[0][1], connection[1][1]], color='purple')
    plt.scatter([point[0] for point in original_points], [point[1] for point in original_points], color='blue', label='Points')
    
    # Last point - start
    plt.plot([chain[-1][0], red_point[0][0]], [chain[-1][1], red_point[0][1]], color='purple')
    plt.scatter(red_point[:, 0], red_point[:, 1], color='red')

    plt.imshow(map_image)
    plt.title('Optimal Rogaining route planner')
    plt.show()


def main():
    # Select a map file
    file_path = filedialog.askopenfilename(title="Select Map Image", filetypes=[("PNG files", "*.png")])
    if file_path:
        map_image = Image.open(file_path)

        root = tk.Tk()
        app = PointSelectionApp(root, map_image)
        root.mainloop()

        # When the window is closed, draw the shortest path
        if app.points:
            draw_shortest_path(app.points, map_image)

if __name__ == "__main__":
    main()
