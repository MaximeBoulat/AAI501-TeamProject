import pandas as pd
import numpy as np
import ast

def compute_angle(from_pos, to_pos):
    dx = to_pos[0] - from_pos[0]
    dy = to_pos[1] - from_pos[1]
    angle = np.arctan2(dy, dx)
    return angle

# Load dataset
df = pd.read_csv("robot_training_data.csv")

# Convert path from string to list of tuples
df['a_star_path'] = df['a_star_path'].apply(ast.literal_eval)  # e.g., "[(2,3), (2,4), (2,5)]" -> [(2,3), (2,4), (2,5)]

# Compute lookahead direction
lookahead_steps = 3
new_directions = []

for path in df['a_star_path']:
    if len(path) >= lookahead_steps + 1:
        current = path[0]
        lookahead = path[lookahead_steps]
        angle = compute_angle(current, lookahead)
    else:
        angle = 0.0  # fallback if path too short
    new_directions.append(angle)

df["goal_direction"] = new_directions
df.to_csv("robot_training_data_with_direction.csv", index=False)

