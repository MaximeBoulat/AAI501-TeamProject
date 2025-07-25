# generate_dataset.py
import numpy as np
import heapq
import csv
import random
import os

WORLD_SIZE = 20
ATTEMPTS = 100
MIN_START_GOAL_DISTANCE = 8
NUM_RUNS = 50
SEED = 42

random.seed(SEED)
np.random.seed(SEED)

DIRECTIONS = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]

def load_world(run_id, input_dir="../data/worlds"):
    return np.load(os.path.join(input_dir, f"world_{run_id}.npy"))

def is_valid(world, x, y):
    return 0 <= x < world.shape[1] and 0 <= y < world.shape[0] and world[y][x] == 0

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
        for dy, dx in DIRECTIONS:
            nx, ny = current[0] + dx, current[1] + dy
            if is_valid(world, nx, ny):
                new_cost = cost + np.linalg.norm([dx, dy])
                heuristic = np.linalg.norm(np.subtract((nx, ny), goal))
                heapq.heappush(open_set, (new_cost + heuristic, new_cost, (nx, ny), path + [current]))
    return []

def get_sensor_readings(world, pos):
    readings = []
    for dy, dx in DIRECTIONS:
        dist = 0
        x, y = pos[0], pos[1]
        while True:
            x += dx
            y += dy
            dist += 1
            if not (0 <= x < world.shape[1] and 0 <= y < world.shape[0]) or world[y][x] == 1:
                break
        readings.append(dist)

    # Direction to goal as angle from north
    gx, gy = goal_pos
    px, py = pos
    dx = gx - px
    dy = py - gy  # y-axis reversed to make 0 degrees point north

    angle_rad = math.atan2(dx, dy)
    angle_deg = (math.degrees(angle_rad) + 360) % 360  # Normalize to [0, 360)

    return readings, angle_deg

def get_action(from_pos, to_pos):
    dx = to_pos[0] - from_pos[0]
    dy = to_pos[1] - from_pos[1]
    for i, (ddy, ddx) in enumerate(DIRECTIONS):
        if (dx, dy) == (ddx, ddy):
            return i
    return None

def simulate_and_generate_data(world, run_id, start_ts):
    for _ in range(ATTEMPTS):
        start = (random.randint(0, WORLD_SIZE - 1), random.randint(0, WORLD_SIZE - 1))
        goal = (random.randint(0, WORLD_SIZE - 1), random.randint(0, WORLD_SIZE - 1))
        if np.linalg.norm(np.subtract(start, goal)) < MIN_START_GOAL_DISTANCE:
            continue
        path = a_star(world, start, goal)
        if path:
            data = []
            ts = start_ts
            for i in range(len(path) - 1):
                sensors = get_sensor_readings(world, path[i])
                action = get_action(path[i], path[i + 1])
                if action is not None:
                    distance = np.linalg.norm(np.subtract(path[-1], path[i]))
                    data.append((ts, run_id, sensors, action, distance))
                    ts += 1
            return data, ts
    return [], start_ts

if __name__ == "__main__":
    all_data = []
    ts = 0
    for run_id in range(NUM_RUNS):
        world = load_world(run_id)
        run_data, ts = simulate_and_generate_data(world, run_id, ts)
        all_data.extend(run_data)

    with open("../data/robot_training_data.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "run_id"] + [f"sensor_{i}" for i in range(8)] + ["action", "distance_to_goal", "angle_to_goal"])
        for ts, run_id, sensors, action, dist in all_data:
            writer.writerow([ts, run_id] + sensors + [action, dist])

