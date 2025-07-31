import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sensors import read_sensors, get_direction_label
from store_map import load_world

X = []
y = []

world_dir = "worlds"
world_files = [f for f in os.listdir(world_dir) if f.endswith(".npy")]

for filename in world_files:
    path = os.path.join(world_dir, filename)
    world = np.load(path)
    
    start_locs = np.argwhere(world == 2)
    goal_locs = np.argwhere(world == 3)
    
    if len(start_locs) == 0 or len(goal_locs) == 0:
        print(f"⚠️ Skipping world {filename}: Missing start or goal")
        continue

    start = tuple(start_locs[0])
    goal = tuple(goal_locs[0])

    # Simple straight-line training path generator
    current = start
    while current != goal:
        sensor_features = read_sensors(world, current, goal)
        # move closer to goal greedily
        dx = np.sign(goal[0] - current[0])
        dy = np.sign(goal[1] - current[1])
        next_pos = (current[0] + dx, current[1] + dy)
        
        label = get_direction_label(current, next_pos)
        X.append(sensor_features)
        y.append(label)
        current = next_pos

print(f"✅ Training on {len(X)} samples...")
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, y)
joblib.dump(clf, "robot_model.joblib")
print("✅ Model saved to robot_model.joblib")
