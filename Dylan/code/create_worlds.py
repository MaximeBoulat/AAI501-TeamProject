# generate_worlds.py
import numpy as np
import os
import random

WORLD_SIZE = 20
WALL_COUNT = 5
WALL_MAX_LEN = 10
OBSTACLE_PROB = 0.2
SEED = 42
NUM_RUNS = 50

random.seed(SEED)
np.random.seed(SEED)

def create_world(size, obstacle_prob, wall_count, max_wall_length):
    world = np.zeros((size, size), dtype=int)
    for y in range(size):
        for x in range(size):
            if random.random() < obstacle_prob:
                world[y][x] = 1
    for _ in range(wall_count):
        is_horizontal = random.choice([True, False])
        length = random.randint(3, max_wall_length)
        if is_horizontal:
            y = random.randint(0, size - 1)
            x_start = random.randint(0, size - length)
            for i in range(length):
                world[y][x_start + i] = 1
        else:
            x = random.randint(0, size - 1)
            y_start = random.randint(0, size - length)
            for i in range(length):
                world[y_start + i][x] = 1
    return world

def save_world(world, run_id, output_dir="../data/worlds"):
    os.makedirs(output_dir, exist_ok=True)

    # Save as .npy
    np.save(os.path.join(output_dir, f"world_{run_id}.npy"), world)

    # Save as human-readable .txt
    with open(os.path.join(output_dir, f"world_{run_id}.txt"), "w") as f:
        for row in world:
            f.write(" ".join(str(cell) for cell in row) + "\n")

if __name__ == "__main__":
    for run_id in range(NUM_RUNS):
        world = create_world(WORLD_SIZE, OBSTACLE_PROB, WALL_COUNT, WALL_MAX_LEN)
        save_world(world, run_id)

