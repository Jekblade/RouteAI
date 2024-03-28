import tkinter as tk
from turtle import color
from PIL import Image, ImageTk
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import itertools

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
             # Always add the point to the canvas
            color = "red" if not self.red_point_selected else "blue"
            point_item = self.canvas.create_oval(x - 10, y - 10, x + 10, y + 10, outline=color, width=2, fill='')
            self.points.append((x, y, point_item))  # Add the point to the list

            # If the red point hasn't been selected yet, mark it as red
            if not self.red_point_selected:
                self.red_point = (x, y)
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


# 1: MST
def minimum_spanning_tree(points):
    points_array = np.array([point for point in points])
    G = nx.Graph()

    for i, (x, y, _) in enumerate(points):
        G.add_node(i, pos=(x, y))

    # Add edges between all pairs of nodes with calculated distances
    for i in range(len(points) - 1):
        for j in range(i + 1, len(points)):
            distance = np.linalg.norm(points_array[i] - points_array[j])
            G.add_edge(i, j, weight=distance)

    if not nx.is_connected(G):
        largest_component = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_component).copy()

    # Compute the minimum spanning tree (connect all points)
    MST = nx.minimum_spanning_tree(G)
    print("\nMinimum spanning tree generated: ", MST)

    return MST



# 2: Odd degree vertices
def find_odd_degree_vertices(G):
    odd_degree_vertices = [v for v, d in G.degree() if d % 2 != 0]
    print("Odd degree vertices: ", odd_degree_vertices)
    return odd_degree_vertices



# 3: Minimum distance nodes matching
def find_min_weight_matching(odd_degree_vertices, G):
    # This uses the full graph G to calculate distances between odd degree vertices
    subgraph = G.subgraph(odd_degree_vertices)
    min_weight_matching = nx.algorithms.matching.max_weight_matching(subgraph, maxcardinality=True, weight='weight')
    print("Minimum weight matching: ", min_weight_matching)

    return min_weight_matching


def tour_length(tour, points):
    # Calculates the total length of the tour based on the points' coordinates
    length = sum(np.linalg.norm(np.array(points[tour[i]]) - np.array(points[tour[i-1]])) for i in range(1, len(tour)))
    return length


# 4: Combine
def combine_graphs(T, M):
    G = nx.Graph(T.edges())
    
    for u, v in M:
        G.add_edge(u, v)
    return G


def generate_eulerian_tour(G):
    is_connected = nx.is_connected(G)
    print("Is the graph connected?", is_connected)

    eulerian_tour = list(nx.eulerian_circuit(G))
    return eulerian_tour


def generate_tsp_tour(eulerian_tour, map_image, points):
    tour = []
    visited = set()
    for u, v in eulerian_tour:
        if u not in visited:
            tour.append(u)
            visited.add(u)
    tour.append(tour[0])
    return tour

def two_opt_swap(tour, i, k):
    return tour[:i] + tour[i:k+1][::-1] + tour[k+1:]

def two_opt(tour, points):
    """Improves the tour by repeatedly applying 2-opt swaps."""
    improvement = True
    while improvement:
        improvement = False
        for i in range(1, len(tour) - 2):
            for k in range(i+1, len(tour)):
                new_tour = two_opt_swap(tour, i, k)
                if tour_length(new_tour, points) < tour_length(tour, points):
                    tour = new_tour
                    improvement = True
    plt.scatter(points[0][0],points[0][1], color="red")
    plt.scatter([point[0] for point in points[1:]], [point[1] for point in points[1:]], color="blue")

    tsp_tour_unique = list(dict.fromkeys(tour))

    # Extract x and y coordinates of the unique tsp_tour vertices
    x = [points[i][0] for i in tsp_tour_unique]
    y = [points[i][1] for i in tsp_tour_unique]
    plt.plot(x, y, marker='o')
    plt.scatter(points[tsp_tour_unique[-1]][0], points[tsp_tour_unique[-1]][1], color='red')  # End point

    plt.title("Optimal Rogaining route using the Christofides algorithm")
    plt.show()





def main():
    file_path = filedialog.askopenfilename(title="Select Map Image", filetypes=[("PNG files", "*.png")])
    if file_path:
        map_image = Image.open(file_path)
        root = tk.Tk()
        app = PointSelectionApp(root, map_image)
        root.mainloop()

        if app.points:

            # Do some shenanigans
            MST = minimum_spanning_tree(app.points)  # Needs implementation
            odd_degree_vertices = find_odd_degree_vertices(MST)
            min_weight_matching = find_min_weight_matching(odd_degree_vertices, MST)  # Adjusted to use MST for distances
            combined_graph = combine_graphs(MST, min_weight_matching)

            eulerian_tour = generate_eulerian_tour(combined_graph)
            tsp_tour = generate_tsp_tour(eulerian_tour, map_image, app.points)
            optimized_tour = two_opt(tsp_tour, app.points)  # Now directly optimizes the TSP tour
            return

if __name__ == "__main__":
    main()