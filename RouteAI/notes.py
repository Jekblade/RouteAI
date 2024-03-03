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

    # Add the starting point
    path.append(start_point)

    return path[::-1], dp[end_point]




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

    # Check if the red point is closer to the last point than the last point's nearest neighbor
    if np.linalg.norm(original_points[-1] - red_point) < distances[0][-1]:
        connections.append((chain[-1], chain[0]))

    # Plot the points and connections
    plt.figure(figsize=(8, 6))
    for connection in connections:
        plt.plot([connection[0][0], connection[1][0]], [connection[0][1], connection[1][1]], color='green', linewidth=3)
    plt.scatter([point[0] for point in original_points], [point[1] for point in original_points], color='blue', label='Points')
    

    def find_minimum_spanning_tree(points):
        points_array = np.array([point[:2] for point in points])
        G = nx.Graph()

        for i, (x, y, _) in enumerate(points):
            G.add_node(i, pos=(x, y))

        for i in range(len(points) - 1):
            for j in range(i + 1, len(points)):
                distance = np.linalg.norm(points_array[i] - points_array[j])
                G.add_edge(i, j, weight=distance)

        MST = nx.minimum_spanning_tree(G)
        return MST

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

def generate_eulerian_tour(G, red_point):
    try:
        eulerian_tour = list(nx.eulerian_circuit(G))
    except:
        # Check if the tour is Eulerian
        if not is_eulerian_tour(eulerian_tour):
            # Find the furthest point from the starting point
            max_distance = -1
            furthest_point = None
            for point in eulerian_tour:
                distance = nx.shortest_path_length(G, red_point, point[0])
                if distance > max_distance:
                    max_distance = distance
                    furthest_point = point[0]
            # Remove the furthest point and generate the new tour
            eulerian_tour = [(u, v) for u, v in eulerian_tour if u != furthest_point and v != furthest_point]
        return eulerian_tour

def is_eulerian_tour(tour):
    # Check if each node appears exactly twice in the tour
    node_count = {}
    for edge in tour:
        if edge[0] not in node_count:
            node_count[edge[0]] = 1
        else:
            node_count[edge[0]] += 1
        if edge[1] not in node_count:
            node_count[edge[1]] = 1
        else:
            node_count[edge[1]] += 1
    for count in node_count.values():
        if count != 2:
            return False
    return True


def generate_tsp_tour(eulerian_tour):
    tsp_tour = [eulerian_tour[i][0] for i in range(len(eulerian_tour))]
    tsp_tour.append(eulerian_tour[0][0])
    return tsp_tour



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
            MST = find_minimum_spanning_tree(app.points)
            odd_degree_vertices = find_odd_degree_vertices(MST)
            min_weight_matching = find_min_weight_matching(MST, odd_degree_vertices)
            combined_graph = combine_graphs(MST, min_weight_matching)
            eulerian_tour = generate_eulerian_tour(combined_graph, app.points[0][:2])
            tsp_tour = generate_tsp_tour(eulerian_tour)

            plt.imshow(map_image)
            points_array = np.array([point[:2] for point in app.points])
            plt.scatter(points_array[:, 0], points_array[:, 1], color='blue')

            # Draw the connections representing the TSP tour
            for i in range(len(tsp_tour) - 1):
                current_point = tsp_tour[i]
                next_point = tsp_tour[i + 1]
                x1, y1, _ = app.points[current_point]
                x2, y2, _ = app.points[next_point]
                plt.plot([x1, x2], [y1, y2], color='red')

            # Connect the last point to the first point to complete the tour
            x1, y1, _ = app.points[tsp_tour[-1]]
            x2, y2, _ = app.points[tsp_tour[0]]
            plt.plot([x1, x2], [y1, y2], color='red')

            plt.title('Optimal rogaining route planning')
            plt.show()
