import numpy as np
import os
import torch
from tqdm import tqdm
from src.astar import solve


def generate_random_map(size=20, obstacle_prob=0.3):
    """
    Generates a 2D grid map.
    0 = Free space
    1 = Obstacle
    """
    # Create a grid of random 0s and 1s
    grid = np.random.choice([0, 1], size=(size, size), p=[1 - obstacle_prob, obstacle_prob])

    # Ensure start (0,0) and goal (size-1, size-1) are always free
    grid[0, 0] = 0
    grid[size - 1, size - 1] = 0

    return grid


def generate_dataset(num_samples, save_path):
    """
    Generates N valid map/path pairs and saves them to a file.
    """
    valid_data = []

    # Configuration (Fixed for now, can be made dynamic later)
    MAP_SIZE = 20
    OBSTACLE_PROB = 0.3
    START = (0, 0)
    GOAL = (MAP_SIZE - 1, MAP_SIZE - 1)

    print(f"Generatng {num_samples} valid samples...")

    # We use a while loop because some maps might be unsolvable,
    # and we want exactly 'num_samples' VALID ones.
    count = 0
    pbar = tqdm(total=num_samples)

    while count < num_samples:
        grid = generate_random_map(MAP_SIZE, OBSTACLE_PROB)
        path = solve(grid, START, GOAL)

        if path is not None:
            # Save the Grid and the Path
            valid_data.append({
                'grid': grid,
                'start': START,
                'goal': GOAL,
                'path': np.array(path)
            })
            count += 1
            pbar.update(1)

    pbar.close()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(valid_data, save_path)
    print(f"✅ Dataset saved to {save_path}")