import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.optimize import linear_sum_assignment

class PointSelectionApp:
    def __init__(self, master, map_image_path):
        self.master = master
        self.master.title("Rogaining course planning with Minimum Spanning Tree")

        self.points = []
        self.red_point = None
        self.red_point_selected = False

        self.canvas = tk.Canvas(master)
        self.canvas.pack(side="left", fill="y")

        self.v_scrollbar = tk.Scrollbar(master, orient="vertical", command=self.canvas.yview)
        self.v_scrollbar.pack(side="right", fill="y")
        self.canvas.configure(yscrollcommand=self.v_scrollbar.set)

        self.map_photo = ImageTk.PhotoImage(Image.open(map_image_path))
        self.image_item = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.map_photo)
        canvas_width = self.map_photo.width()
        self.canvas.config(width=canvas_width)

        self.canvas.bind("<Button-1>", self.add_point)
        self.master.bind("<Return>", self.complete_selection)

        self.canvas.update_idletasks()
        self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))

        self.map_width = self.map_photo.width()
        
        if self.map_height < master.winfo_screenheight():
            self.map_height = master.winfo_screenheight()
        else:
            self.map_height = self.map_photo.height()

    def add_point(self, event):
        x, y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        x += self.canvas.xview()[0]

        if 0 <= x <= self.map_width and 0 <= y <= self.map_height:
            if not self.red_point_selected:
                self.red_point = (x, y)
            color = "red" if not self.red_point_selected else "blue"
            self.red_point_selected = True
            point_item = self.canvas.create_oval(x-6, y-6, x+6, y+6, fill=color, outline=color)
            self.points.append((x, y, point_item))

    def complete_selection(self, event):
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

    MST = nx.minimum_spanning_tree(G)

    plt.scatter(points_array[:, 0], points_array[:, 1], color='blue')
    for u, v in MST.edges():
        x1, y1, _ = points[u]
        x2, y2, _ = points[v]
        plt.plot([x1, x2], [y1, y2], color='black')

    plt.scatter(red_point[0], red_point[1], color='red')
    plt.imshow(map_image)
    plt.title('Minimum Spanning Tree with Red Point')
    plt.show()

def find_odd_degree_vertices(G):
    odd_degree_vertices = [v for v, d in G.degree if d % 2 != 0]
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
    map_image_path = "map.jpg"
    root = tk.Tk()
    app = PointSelectionApp(root, map_image_path)
    root.mainloop()

    if app.points:
        G = create_graph(app.points)
        red_point_index = app.points.index(app.red_point)
        odd_degree_vertices = find_odd_degree_vertices(G)
        odd_degree_vertices.remove(red_point_index)  # Remove red point from odd-degree vertices
        min_weight_matching = find_min_weight_matching(G, odd_degree_vertices)
        combined_graph = combine_graphs(G, min_weight_matching)
        eulerian_tour = generate_eulerian_tour(combined_graph)
        tsp_tour = generate_tsp_tour(eulerian_tour)
        minimum_spanning_tree(app.points, app.red_point, Image.open(map_image_path))

if __name__ == "__main__":
    main()
