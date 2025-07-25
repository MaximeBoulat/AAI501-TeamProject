# print_saved_worlds.py

import numpy as np
import os

WORLD_DIR = "../data/worlds"
NUM_WORLDS = 50  # or however many you generated

def print_world_only(world):
    display = np.full(world.shape, ".", dtype=str)
    for y in range(world.shape[0]):
        for x in range(world.shape[1]):
            if world[y][x] == 1:
                display[y][x] = "#"
    for row in display:
        print(" ".join(row))

def load_and_print_world(run_id, directory=WORLD_DIR):
    path = os.path.join(directory, f"world_{run_id}.npy")
    if not os.path.exists(path):
        print(f"World {run_id} not found at {path}")
        return
    world = np.load(path)
    print(f"\n=== World {run_id} ===")
    print_world_only(world)

if __name__ == "__main__":
    for run_id in range(NUM_WORLDS):
        load_and_print_world(run_id)

