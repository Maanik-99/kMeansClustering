import numpy as np
import random

# --- Configuration ---
NUM_POINTS = 100
NUM_CLUSTERS = 10
MAX_COORD_VAL = 30  # Max coordinate value for x and y
GRID_WIDTH = MAX_COORD_VAL + 1
GRID_HEIGHT = MAX_COORD_VAL + 1
NUM_ITERATIONS = 10
DATA_POINTS_FILE = "data_points.txt"
INITIAL_CENTROIDS_FILE = "initial_centroids.txt"

# --- Function Definitions ---

def generate_data(num_p, num_c, max_val):
    """Generates random 2D data points and initial centroids, saves them to files."""
    points = np.random.randint(0, max_val + 1, size=(num_p, 2))
    centroids = np.random.randint(0, max_val + 1, size=(num_c, 2))
    
    np.savetxt(DATA_POINTS_FILE, points, fmt='%d', delimiter=',', header='x,y', comments='')
    np.savetxt(INITIAL_CENTROIDS_FILE, centroids, fmt='%d', delimiter=',', header='x,y', comments='')
    
    print(f"Generated {len(points)} data points, saved to {DATA_POINTS_FILE}")
    print(f"Generated {len(centroids)} initial centroids, saved to {INITIAL_CENTROIDS_FILE}")
    return points, centroids

def manhattan_distance(p1, p2):
    """Calculates Manhattan distance between two 2D points."""
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

def assign_to_clusters(data_points, centroids):
    """Assigns each data point to the nearest cluster based on Manhattan distance."""
    assignments = np.zeros(len(data_points), dtype=int)
    for i, point in enumerate(data_points):
        distances = [manhattan_distance(point, centroid) for centroid in centroids]
        assignments[i] = np.argmin(distances)
    return assignments

def update_centroids(data_points, assignments, num_clusters, current_centroids):
    """Updates centroids to be the mean of points assigned to them."""
    new_centroids = np.copy(current_centroids)
    for k in range(num_clusters):
        points_in_cluster = data_points[assignments == k]
        if len(points_in_cluster) > 0:
            new_centroids[k] = np.mean(points_in_cluster, axis=0)
        else:
            print(f"Warning: Cluster {k} became empty. Re-initializing its centroid randomly.")
            new_centroids[k] = data_points[np.random.choice(len(data_points))]
    return new_centroids

def k_means_manhattan(data_points, initial_centroids, iterations):
    """Performs K-Means clustering using Manhattan distance."""
    centroids = initial_centroids.astype(float)
    
    for i in range(iterations):
        print(f"Iteration {i+1}/{iterations}")
        assignments = assign_to_clusters(data_points, centroids)
        new_centroids = update_centroids(data_points, assignments, len(centroids), centroids)
        
        if np.allclose(centroids, new_centroids):
            print(f"Converged at iteration {i+1}.")
            break
        centroids = new_centroids
        
    final_assignments = assign_to_clusters(data_points, centroids)
    return centroids, final_assignments

def create_visualization_matrix(data_points, final_centroids, assignments, width, height):
    """Creates a 2D character matrix for visualization."""
    grid = [['.' for _ in range(width)] for _ in range(height)]
    
    point_symbols = [str(i) for i in range(10)] 
    centroid_symbols = [chr(ord('A') + i) for i in range(10)]

    for i, point in enumerate(data_points):
        x = min(width - 1, max(0, int(round(point[0]))))
        y = min(height - 1, max(0, int(round(point[1]))))
        cluster_idx = assignments[i]
        grid[y][x] = point_symbols[cluster_idx % len(point_symbols)]

    for i, centroid in enumerate(final_centroids):
        x = min(width - 1, max(0, int(round(centroid[0]))))
        y = min(height - 1, max(0, int(round(centroid[1]))))
        grid[y][x] = centroid_symbols[i % len(centroid_symbols)]
            
    return grid

def print_visualization(matrix, width, height):
    """Prints the 2D visualization matrix to the console."""
    print("\nClustering Visualization (Manhattan K-Means):")
    print("Data points: 0-9 (cluster ID), Centroids: A-J (cluster ID)")
    
    tens_header_list = [' '] * width
    units_header_list = [' '] * width
    for i in range(width):
        tens_header_list[i] = str(i // 10) if i // 10 != 0 else ' '
        units_header_list[i] = str(i % 10)
    
    print("   " + "".join(tens_header_list))
    print("   " + "".join(units_header_list))
    print("  +" + "-" * width + "+")

    for r_idx in range(height -1, -1, -1):
        row_str = "".join(matrix[r_idx])
        print(f"{r_idx:2d}|{row_str}|")
    print("  +" + "-" * width + "+")
    print("\nNote: Y-axis is printed from max value down to 0.")

# Main execution block for demonstration
if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)

    print("--- K-Means Clustering with Manhattan Distance ---")
    
    print("\n--- 1. Data Generation ---")
    data_p, initial_c = generate_data(NUM_POINTS, NUM_CLUSTERS, MAX_COORD_VAL)
    
    print("\nSample Initial Centroids (first 3):")
    print(initial_c[:3])
    print("\nSample Data Points (first 3):")
    print(data_p[:3])
    
    print("\n--- 2. Running K-Means Algorithm ---")
    final_centroids, final_assignments = k_means_manhattan(data_p, initial_c, NUM_ITERATIONS)
    
    print("\nFinal Centroids:")
    for i, fc in enumerate(final_centroids):
        print(f"Cluster {chr(ord('A')+i)} ({i}): ({fc[0]:.2f}, {fc[1]:.2f})")
    
    print("\n--- 3. Visualization ---")
    vis_matrix = create_visualization_matrix(data_p, final_centroids, final_assignments, GRID_WIDTH, GRID_HEIGHT)
    print_visualization(vis_matrix, GRID_WIDTH, GRID_HEIGHT)
    
    print("\n--- Experiment Complete ---")
