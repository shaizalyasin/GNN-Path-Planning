import heapq
import numpy as np


def heuristic(a, b):
    """
    Calculates the Manhattan distance between two points a and b.
    (x1, y1) and (x2, y2).
    """
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def solve(grid, start, goal):
    """
    Finds the shortest path on a 2D grid using the A* algorithm.

    Args:
        grid (numpy.ndarray): 2D array where 0 is free space and 1 is an obstacle.
        start (tuple): (row, col) coordinates of the start point.
        goal (tuple): (row, col) coordinates of the goal point.

    Returns:
        list: A list of tuples [(r, c), ...] representing the path from start to goal.
              Returns None if no path is found.
    """
    rows, cols = grid.shape

    # Priority Queue
    open_set = []
    heapq.heappush(open_set, (0, start))

    came_from = {}

    # Cost from start to current node
    g_score = {node: float('inf') for node in np.ndindex(rows, cols)}
    g_score[start] = 0

    # Estimated total cost
    f_score = {node: float('inf') for node in np.ndindex(rows, cols)}
    f_score[start] = heuristic(start, goal)

    open_set_hash = {start}

    while open_set:
        # Pop the node with the lowest f_score
        current_f, current = heapq.heappop(open_set)
        open_set_hash.remove(current)

        # 1. Check if we reached the goal
        if current == goal:
            return reconstruct_path(came_from, current)

        # 2. Check neighbors (Up, Down, Left, Right)
        neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        for dx, dy in neighbors:
            neighbor = (current[0] + dx, current[1] + dy)

            # Check boundaries
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols:
                # Check if it's an obstacle
                if grid[neighbor] == 1:
                    continue

                tentative_g_score = g_score[current] + 1

                # If we found a better path to this neighbor
                if tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)

                    if neighbor not in open_set_hash:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
                        open_set_hash.add(neighbor)

    return None


def reconstruct_path(came_from, current):
    """
    Backtracks from the goal to the start to rebuild the path.
    """
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    return path[::-1]