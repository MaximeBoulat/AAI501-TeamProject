import numpy as np
import os

WORLD_SIZE = (20, 20)
NUM_WORLDS = 50
OBSTACLE_PROB = 0.2
SAVE_DIR = "worlds"

os.makedirs(SAVE_DIR, exist_ok=True)

for i in range(NUM_WORLDS):
    world = np.zeros(WORLD_SIZE, dtype=int)

    # Add obstacles randomly
    mask = np.random.rand(*WORLD_SIZE) < OBSTACLE_PROB
    world[mask] = 1

    # Random start and goal positions on free tiles
    free_positions = list(zip(*np.where(world == 0)))
    if len(free_positions) < 2:
        print(f"❌ Skipping world {i}: Not enough free space.")
        continue

    start, goal = np.random.choice(len(free_positions), 2, replace=False)
    sx, sy = free_positions[start]
    gx, gy = free_positions[goal]

    world[sx, sy] = 2  # start
    world[gx, gy] = 3  # goal

    np.save(os.path.join(SAVE_DIR, f"world_{i}.npy"), world)

    # Optional: Save human-readable version
    np.savetxt(os.path.join(SAVE_DIR, f"world_{i}.txt"), world, fmt='%d')

print("✅ World generation complete.")
