# ✅ main_simulation.py
import joblib
import numpy as np
from store_map import store_grid
from robot_agent import InventoryRobot  # ✅ Correct class name
from sensors import read_sensors

# Load trained model
clf = joblib.load("best_mlp_model.pkl")

# Find start and goal tiles safely
start_coords = np.argwhere(store_grid == 2)
goal_coords = np.argwhere(store_grid == 3)

if len(start_coords) == 0 or len(goal_coords) == 0:
    raise ValueError("Start tile (2) or goal tile (3) not found in the store_grid.")

start = tuple(start_coords[0])
goal = tuple(goal_coords[0])

# Run simulation
robot = InventoryRobot(world=store_grid, start=start, goal=goal, model=clf)
robot.navigate(max_steps=100)

# Final path
print("\n✅ Final path taken:")
print(robot.path)

