import numpy as np
import heapq
import random
import csv
import os
import math

from world import World
from config import *

# Set random seed for built-in random module
random.seed(SEED)

# Set random seed for NumPy
np.random.seed(SEED)

#0 = left, 1 = left+down, 2 = down, 3=right+down, 4=right, 5=right+up,  6=up,   7=left+up 
DIRECTIONS = [(-1, 0),   (-1, 1),      (0, 1),   (1, 1),        (1, 0),  (1, -1),    (0, -1), (-1, -1)]



def is_valid(world, x, y):
    return 0 <= x < world.size and 0 <= y < world.size and world.grid[y][x] == 0

def a_star(world, start, goal):
    open_set = []
    heapq.heappush(open_set, (0 + np.linalg.norm(np.subtract(start, goal)), 0, start, []))
    visited = set()

    while open_set:
        _, cost, current, path = heapq.heappop(open_set)

        if current in visited:
            continue
        visited.add(current)

        if current == goal:
            return path + [current]

        for i, (dy, dx) in enumerate(DIRECTIONS):
            nx, ny = current[0] + dx, current[1] + dy
            if is_valid(world, nx, ny):
                new_cost = cost + np.linalg.norm([dx, dy])
                heuristic = np.linalg.norm(np.subtract((nx, ny), goal))
                heapq.heappush(open_set, (new_cost + heuristic, new_cost, (nx, ny), path + [current]))

    return []


def get_action(from_pos, to_pos):
    dx = to_pos[0] - from_pos[0]
    dy = to_pos[1] - from_pos[1]
    for i, (ddx, ddy) in enumerate(DIRECTIONS):
        if (dx, dy) == (ddx, ddy):
            return i
    return None

def get_goal_direction_index(current, goal):
    dx = goal[0] - current[0]
    dy = goal[1] - current[1]
    if dx == 0 and dy == 0:
        return None  # At goal
    # Normalize to unit direction
    norm_dx = np.sign(dx)
    norm_dy = np.sign(dy)
    for i, (ddx, ddy) in enumerate(DIRECTIONS):
        if (norm_dx, norm_dy) == (ddx, ddy):
            return i
    return None

def get_goal_direction_radians(current, goal):
    dx = goal[0] - current[0]
    dy = current[1] - goal[1]  # Invert Y to make 0 point north

    angle = math.atan2(dx, dy)  # dx first because 0° = north
    angle = angle % (2 * math.pi) / (2 * math.pi) # Normalize to [0, 1)
    return angle

def generate_training_data(world, path, run_id, starting_timestamp):
    data = []
    timestamp = starting_timestamp
    for i in range(len(path) - 1):
        sensors = world.get_sensor_readings(path[i])
        action = get_action(path[i], path[i + 1])
        if action is not None:
            remaining = np.linalg.norm(np.subtract(path[-1], path[i]))
            goal_direction = get_goal_direction_radians(path[i], path[-1])
            data.append((timestamp, run_id, sensors, action, remaining, goal_direction))
            timestamp += 1
    return data, timestamp

def print_world(world, path, start, goal):
    display = np.full(world.grid.shape, ".", dtype=str)
    for y in range(world.size):
        for x in range(world.size):
            if world.grid[y][x] == 1:
                display[y][x] = "#"
    for (x, y) in path:
        if (x, y) != start and (x, y) != goal:
            display[y][x] = "*"
    sx, sy = start
    gx, gy = goal
    display[sy][sx] = "S"
    display[gy][gx] = "G"
    for row in display:
        print(" ".join(row))

def save_world_to_disk(world, run_id, output_dir="worlds"):
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as .npy (binary format)
    np.save(os.path.join(output_dir, f"world_{run_id}.npy"), world.grid)
    
    # Optional: Save as text for inspection
    with open(os.path.join(output_dir, f"world_{run_id}.txt"), "w") as f:
        for row in world.grid:
            f.write(" ".join(str(cell) for cell in row) + "\n")

def simulate_world(run_id, starting_timestamp):
    attempts = 0
    while attempts < ATTEMPTS:
        world = World.from_random(WORLD_SIZE, OBSTACLE_PROB, WALL_COUNT, WALL_MAX_LEN, MIN_START_GOAL_DISTANCE)
        start = (random.randint(0, WORLD_SIZE - 1), random.randint(0, WORLD_SIZE - 1))
        goal = (random.randint(0, WORLD_SIZE - 1), random.randint(0, WORLD_SIZE - 1))

        # Check distance is significant to make it interesting
        dist = np.linalg.norm(np.subtract(start, goal))
        if not (MIN_START_GOAL_DISTANCE <= dist):
            attempts += 1
            continue
        
        path = a_star(world, start, goal)
        if path:
            data, next_timestamp = generate_training_data(world, path, run_id, starting_timestamp)
            save_world_to_disk(world, run_id)
            print(f"\n=== World Map: Run {run_id} ===")
            print_world(world, path, start, goal)
            return data, next_timestamp
        attempts += 1
    return [], starting_timestamp

# === Run Multiple Worlds ===
global_timestamp = 0
all_training_data = []

for run_id in range(NUM_RUNS):
    run_data, global_timestamp = simulate_world(run_id, global_timestamp)
    all_training_data.extend(run_data)

# === Save All Runs to CSV ===
with open(TRAINING_DATA_FILENAME, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "timestamp", "run_id",
        "sensor_0", "sensor_1", "sensor_2", "sensor_3",
        "sensor_4", "sensor_5", "sensor_6", "sensor_7",
        "action", "distance_to_goal", "goal_direction"
    ])
    for timestamp, run_id, sensors, action, distance_to_goal, goal_direction in all_training_data:
        writer.writerow([timestamp, run_id] + sensors + [action, distance_to_goal, goal_direction])

