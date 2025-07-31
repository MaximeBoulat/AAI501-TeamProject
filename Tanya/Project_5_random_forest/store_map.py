# store_map.py
import numpy as np

# Predefined static map (used for testing or visualization)
store_grid = np.array([
    # 0 = free, 1 = wall/obstacle, 2 = start, 3 = goal
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 2, 1, 0, 1, 0, 1, 0, 1, 1],  # Start at (1,1)
    [1, 0, 1, 0, 1, 0, 1, 0, 1, 1],
    [1, 0, 1, 0, 1, 0, 1, 0, 1, 1],
    [1, 0, 1, 0, 1, 0, 1, 0, 1, 1],
    [1, 0, 1, 0, 1, 0, 1, 3, 1, 1],  # Goal at (5,7)
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Exit row
])

# ✅ Add this so the training script can load from file
def load_world(filename):
    return np.loadtxt(filename, dtype=int)


