import pandas as pd
import numpy as np
import heapq

# Load dataset
df = pd.read_csv("robot_training_data.csv")

# Define movement directions: (dy, dx) and action codes
directions = {
    (0, -1): 0,   # N
    (1, -1): 1,   # NE
    (1, 0): 2,    # E
    (1, 1): 3,    # SE
    (0, 1): 4,    # S
    (-1, 1): 5,   # SW
    (-1, 0): 6,   # W
    (-1, -1): 7   # NW
}

reverse_directions = {v: k for k, v in directions.items()}

def heuristic(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def astar(grid, start, goal):
    rows, cols = grid.shape
    open_set = [(heuristic(start, goal), 0, start, [])]
    visited = set()

    while open_set:
        est_total, cost, current, path = heapq.heappop(open_set)

        if current in visited:
            continue
        visited.add(current)

        path = path + [current]

        if current == goal:
            return path

        for move, action in directions.items():
            ny, nx = current[0] + move[0], current[1] + move[1]
            if 0 <= ny < rows and 0 <= nx < cols and grid[ny][nx] == 0:
                heapq.heappush(open_set, (cost + heuristic((ny, nx), goal), cost + 1, (ny, nx), path))

    return []

def generate_grid_from_sensors(sensor_row):
    # Create 3x3 grid centered on robot (robot at 1,1)
    grid = np.zeros((3, 3), dtype=int)
    sensor_values = sensor_row[2:10].values  # sensor_0 to sensor_7

    for i, val in enumerate(sensor_values):
        dy, dx = reverse_directions[i]
        y, x = 1 + dy, 1 + dx
        if val > 5:  # Arbitrary threshold: sensor detects obstacle if value > 5
            grid[y][x] = 1

    grid[1][1] = 0  # Ensure center is open (robot)
    return grid

def get_astar_action(row):
    grid = generate_grid_from_sensors(row)
    start = (1, 1)
    goal = (0, 1)  # Placeholder goal: one tile north of robot
    path = astar(grid, start, goal)
    if len(path) >= 2:
        move = (path[1][0] - path[0][0], path[1][1] - path[0][1])
        return directions.get(move, -1)
    return -1

# Generate A* action labels
df['astar_action'] = df.apply(get_astar_action, axis=1)

# Save to new CSV
df.to_csv("robot_data_with_astar_labels.csv", index=False)
print("✅ Saved: robot_data_with_astar_labels.csv")
