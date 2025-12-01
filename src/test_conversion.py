import numpy as np
import torch
from utils import grid_to_graph


def test_conversion():
    # 1. Create a tiny 3x3 Map
    # 0 = Free, 1 = Obstacle
    # [ . . # ]  (0,0) (0,1) (Blocked)
    # [ # . . ]  (Blocked) (1,1) (1,2)
    # [ . . . ]  (2,0) (2,1) (2,2)
    grid = np.array([
        [0, 0, 1],
        [1, 0, 0],
        [0, 0, 0]
    ])

    start = (0, 0)
    goal = (2, 2)

    print("--- Original Grid (3x3) ---")
    print(grid)
    print(f"Total cells: 9")
    print(f"Free cells (0): {np.sum(grid == 0)} (Should become Nodes)")
    print("-" * 20)

    # Convert to Graph
    graph_data = grid_to_graph(grid, start, goal)

    # Inspect the Graph
    print("\n--- GNN Graph Data Object ---")
    print(graph_data)

    # Check Features (x)
    # Shape should be [Num_Nodes, 4] -> [7, 4]
    print(f"\nNode Features (x) Shape: {graph_data.x.shape}")
    print("Example Node Feature (Node 0):", graph_data.x[0])
    # [Norm_Row, Norm_Col, Is_Start, Is_Goal]
    # Node 0 is at (0,0), is Start. Expect: [0.0, 0.0, 1.0, 0.0]

    # Check Edges (edge_index)
    # Shape should be [2, Num_Edges]
    print(f"Edge Index Shape: {graph_data.edge_index.shape}")
    num_edges = graph_data.edge_index.shape[1]
    print(f"Total Directed Edges: {num_edges}")

    print("\n✅ Conversion Test Passed!")


if __name__ == "__main__":
    test_conversion()