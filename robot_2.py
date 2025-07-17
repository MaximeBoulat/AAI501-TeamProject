import numpy as np
import heapq
import random
import csv
import os

num_runs = 50
WORLD_SIZE = 20
ATTEMPTS = 100
OBSTALCE_PROB = 0.2
DIRECTIONS = [(-1, 0), (-1, 1), (0, 1), (1, 1),
              (1, 0), (1, -1), (0, -1), (-1, -1)]

def create_world(size=20, obstacle_prob=0.2, wall_count=5, max_wall_length=10):
    world = np.zeros((size, size), dtype=int)
    
    # Random single-tile obstacles
    for y in range(size):
        for x in range(size):
            if random.random() < obstacle_prob:
                world[y][x] = 1

    # Add horizontal or vertical walls
    for _ in range(wall_count):
        is_horizontal = random.choice([True, False])
        wall_length = random.randint(3, max_wall_length)
        
        if is_horizontal:
            y = random.randint(0, size - 1)
            x_start = random.randint(0, size - wall_length)
            for i in range(wall_length):
                world[y][x_start + i] = 1
        else:
            x = random.randint(0, size - 1)
            y_start = random.randint(0, size - wall_length)
            for i in range(wall_length):
                world[y_start + i][x] = 1

    return world


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

        for i, (dy, dx) in enumerate(DIRECTIONS):
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
    return readings

def get_action(from_pos, to_pos):
    dx = to_pos[0] - from_pos[0]
    dy = to_pos[1] - from_pos[1]
    for i, (ddy, ddx) in enumerate(DIRECTIONS):
        if (dx, dy) == (ddx, ddy):
            return i
    return None

def generate_training_data(world, path, run_id, starting_timestamp):
    data = []
    timestamp = starting_timestamp
    for i in range(len(path) - 1):
        sensors = get_sensor_readings(world, path[i])
        action = get_action(path[i], path[i + 1])
        if action is not None:
            data.append((timestamp, run_id, sensors, action))
            timestamp += 1
    return data, timestamp

def print_world(world, path, start, goal):
    display = np.full(world.shape, ".", dtype=str)
    for y in range(world.shape[0]):
        for x in range(world.shape[1]):
            if world[y][x] == 1:
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
    np.save(os.path.join(output_dir, f"world_{run_id}.npy"), world)
    
    # Optional: Save as text for inspection
    with open(os.path.join(output_dir, f"world_{run_id}.txt"), "w") as f:
        for row in world:
            f.write(" ".join(str(cell) for cell in row) + "\n")

def simulate_world(run_id, starting_timestamp):
    attempts = 0
    while attempts < ATTEMPTS:
        world = create_world(size=WORLD_SIZE, obstacle_prob=OBSTALCE_PROB)
        start = (random.randint(0, WORLD_SIZE - 1), random.randint(0, WORLD_SIZE - 1))
        goal = (random.randint(0, WORLD_SIZE - 1), random.randint(0, WORLD_SIZE - 1))
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

for run_id in range(num_runs):
    run_data, global_timestamp = simulate_world(run_id, global_timestamp)
    all_training_data.extend(run_data)

# === Save All Runs to CSV ===
with open("robot_training_data.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["timestamp", "run_id", "sensor_0", "sensor_1", "sensor_2", "sensor_3",
                     "sensor_4", "sensor_5", "sensor_6", "sensor_7", "action"])
    for timestamp, run_id, sensors, action in all_training_data:
        writer.writerow([timestamp, run_id] + sensors + [action])

