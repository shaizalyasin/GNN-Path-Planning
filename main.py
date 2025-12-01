import argparse
import os
from src.generator import generate_random_map, generate_dataset
from src.astar import solve
from src.utils import plot_map


def run_demo():
    MAP_SIZE = 20
    START = (0, 0)
    GOAL = (MAP_SIZE - 1, MAP_SIZE - 1)
    OBSTACLE_PROB = 0.3
    print(f"--- Running Single Demo (Size: {MAP_SIZE}x{MAP_SIZE}) ---")
    grid = generate_random_map(MAP_SIZE, OBSTACLE_PROB)
    print("Finding optimal path...")
    path = solve(grid, START, GOAL)
    if path:
        print(f"✅ Path found! Length: {len(path)}")
        plot_map(grid, path, START, GOAL, title="Demo: A* Path")
    else:
        print("❌ No path found (Blocked).")
        plot_map(grid, None, START, GOAL, title="Demo: Blocked")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='demo', choices=['demo', 'generate', 'train'])
    # Add a new argument for how many samples to generate
    parser.add_argument('--samples', type=int, default=1000, help="Number of samples to generate")

    args = parser.parse_args()

    if args.mode == 'demo':
        run_demo()

    elif args.mode == 'generate':
        save_file = os.path.join('data', 'dataset_v1.pt')
        generate_dataset(num_samples=args.samples, save_path=save_file)

    elif args.mode == 'train':
        print("Training not implemented yet.")