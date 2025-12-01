import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.data import Data


def plot_map(grid, path=None, start=None, goal=None, title="Grid Map"):
    """
    Visualizes the binary grid, start/goal positions, and the A* path.
    """
    plt.figure(figsize=(6, 6))

    # Draw the grid
    plt.imshow(grid, cmap='binary', origin='upper')

    if path is not None and len(path) > 0:
        # If it's a NumPy array, convert to list for easy zipping
        path_list = path.tolist() if isinstance(path, np.ndarray) else path

        # Unzip path tuples into Y (rows) and X (cols)
        y_coords, x_coords = zip(*path_list)
        plt.plot(x_coords, y_coords, color='red', linewidth=3, label='Path')

    # Draw Start Node (Green)
    if start:
        plt.scatter(start[1], start[0], color='green', s=150, zorder=5, label='Start')

    # Draw Goal Node (Blue)
    if goal:
        plt.scatter(goal[1], goal[0], color='blue', s=150, zorder=5, label='Goal')

    plt.title(title)
    plt.legend(loc='upper right')
    plt.grid(which='both', color='lightgrey', linewidth=0.5)
    plt.xticks([])
    plt.yticks([])

    plt.show()


def grid_to_graph(grid, start, goal):
    """
    Converts a 2D grid into a PyTorch Geometric Graph (Data object).

    Nodes: All free cells (0).
    Edges: Connections between adjacent free cells.
    Features: [Row, Col, Is_Start, Is_Goal]
    """
    rows, cols = grid.shape

    # 1. Identify all free nodes
    # node_indices[i, j] will store the ID (0, 1, 2...) of the node at (i, j)
    # If obstacle, it stays -1
    node_indices = np.full((rows, cols), -1, dtype=int)

    node_features = []
    current_id = 0

    for r in range(rows):
        for c in range(cols):
            if grid[r, c] == 0:  # If free space
                node_indices[r, c] = current_id

                # Feature Vector: [Row Normalised, Col Normalised, IsStart, IsGoal]
                is_start = 1.0 if (r, c) == start else 0.0
                is_goal = 1.0 if (r, c) == goal else 0.0

                # We normalize coordinates to 0-1 range to help the AI learn better
                norm_r = r / rows
                norm_c = c / cols

                node_features.append([norm_r, norm_c, is_start, is_goal])
                current_id += 1

    # 2. Build Edges
    edge_sources = []
    edge_targets = []

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for r in range(rows):
        for c in range(cols):
            src_id = node_indices[r, c]

            # If this is a valid node (not an obstacle)
            if src_id != -1:
                for dr, dc in directions:
                    nr, nc = r + dr, c + dc

                    # Check bounds
                    if 0 <= nr < rows and 0 <= nc < cols:
                        tgt_id = node_indices[nr, nc]

                        # If neighbor is also valid
                        if tgt_id != -1:
                            # Add edge (Source -> Target)
                            edge_sources.append(src_id)
                            edge_targets.append(tgt_id)

    # 3. Convert to PyTorch Tensors
    x = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long)

    # Create Data Object
    data = Data(x=x, edge_index=edge_index)

    # Store mapping for later (so we can reconstruct the path)
    data.node_indices = node_indices

    return data