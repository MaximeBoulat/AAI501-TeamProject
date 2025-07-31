# simulate_robot.py
import numpy as np
import os
import random
import json
import joblib
from inventory_robot import InventoryRobot
from sensors import DIRECTIONS

def get_random_position(world):
    free = np.argwhere(world == 0)
    return tuple(random.choice(free))

clf = joblib.load('robot_model.joblib')
results = []
world_dir = "worlds"
success = 0
failure = 0

for fname in sorted(os.listdir(world_dir)):
    if fname.endswith(".npy"):
        world = np.load(os.path.join(world_dir, fname))
        start = get_random_position(world)
        goal = get_random_position(world)
        while goal == start:
            goal = get_random_position(world)

        robot = InventoryRobot(world, start, goal, clf)
        reached = robot.navigate(max_steps=100)
        results.append({
            "world": fname,
            "start": start,
            "goal": goal,
            "success": reached,
            "steps": len(robot.path)
        })

        if reached:
            success += 1
        else:
            failure += 1

print("\n========== SUMMARY ==========")
print(f"✅ Successes: {success}")
print(f"❌ Failures: {failure}")
print(f"📊 Total: {len(results)}")

with open("navigation_report.json", "w") as f:
    json.dump(results, f, indent=2)
