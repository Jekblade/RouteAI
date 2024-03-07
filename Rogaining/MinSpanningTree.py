import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

class PointSelectionApp:
    def __init__(self, master, map_image):
        self.master = master
        self.master.title("Rogaining course planning with kNearestNeighbour")

        self.points = []
        self.red_point_selected = False
        self.red_point = None

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


def minimum_spanning_tree(points, red_point, map_image):
    points_array = np.array([point[:2] for point in points])
    G = nx.Graph()

    for i, (x, y, _) in enumerate(points):
        G.add_node(i, pos=(x, y))

    for i in range(len(points) - 1):
        for j in range(i + 1, len(points)):
            distance = np.linalg.norm(points_array[i] - points_array[j])
            G.add_edge(i, j, weight=distance)

    # Ensure that the graph is connected
    if not nx.is_connected(G):
        # If the graph is not connected, get the largest connected component
        largest_component = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_component).copy()

    MST = nx.minimum_spanning_tree(G)
    return MST


def find_odd_degree_vertices(G):
    odd_degree_vertices = [v for v, d in G.degree() if d % 2 != 0]
    return odd_degree_vertices

def find_min_weight_matching(G, vertices):
    subgraph = G.subgraph(vertices)
    min_weight_matching = nx.algorithms.matching.max_weight_matching(subgraph, weight='weight')
    return min_weight_matching

def combine_graphs(T, M):
    G = nx.MultiGraph(T)
    for edge in M:
        G.add_edge(*edge)
    return G

def generate_eulerian_tour(G):
    eulerian_tour = list(nx.eulerian_circuit(G))
    return eulerian_tour

def generate_tsp_tour(eulerian_tour):
    tsp_tour = [eulerian_tour[i][0] for i in range(len(eulerian_tour))]
    tsp_tour.append(eulerian_tour[0][0])
    return tsp_tour

def main():
    file_path = filedialog.askopenfilename(title="Select Map Image", filetypes=[("PNG files", "*.png")])
    if file_path:
        map_image = Image.open(file_path)
        root = tk.Tk()
        app = PointSelectionApp(root, map_image)
        root.mainloop()

        if app.points:
            if app.red_point in app.points:
                red_point_index = app.points.index(app.red_point)
                odd_degree_vertices.remove(red_point_index)

            G = minimum_spanning_tree(app.points, app.red_point, map_image)
            
            if not nx.is_connected(G):
                # Connect disconnected components
                components = list(nx.connected_components(G))
                for i in range(len(components) - 1):
                    # Connect the components by adding an edge between arbitrary vertices
                    u = list(components[i])[0]
                    v = list(components[i + 1])[0]
                    G.add_edge(u, v)

            # Ensure all vertices have even degrees
            odd_degree_vertices = find_odd_degree_vertices(G)
            for vertex in odd_degree_vertices:
                # Find a vertex with odd degree
                neighbors = list(G.neighbors(vertex))
                if len(neighbors) % 2 == 0:
                    # If the number of neighbors is even, add an edge to one of them
                    target_vertex = neighbors[0]  # Choose one of the neighbors arbitrarily
                    G.add_edge(vertex, target_vertex)
                else:
                    # If the number of neighbors is odd, add a new vertex and connect it to the current vertex
                    new_vertex = max(G.nodes) + 1  # Generate a new vertex label
                    G.add_node(new_vertex)
                    G.add_edge(vertex, new_vertex)


            min_weight_matching = find_min_weight_matching(G, odd_degree_vertices)
            combined_graph = combine_graphs(G, min_weight_matching)
            eulerian_tour = generate_eulerian_tour(combined_graph)
            tsp_tour = generate_tsp_tour(eulerian_tour)
            minimum_spanning_tree(app.points, app.red_point, map_image)

if __name__ == "__main__":
    main()
