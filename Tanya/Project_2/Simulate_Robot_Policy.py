# Simulate Robot Policy
import pandas as pd
import numpy as np
import joblib  # For loading trained model (e.g., Random Forest)

# === Load the trained model ===
# Replace with your model path as needed
model = joblib.load("classifier_outputs/random_forest_model.pkl")  # or neural_net_model.h5 if using keras

# === Load the labeled dataset ===
df = pd.read_csv("robot_data_with_astar_labels.csv")

# === Simulate one full run ===
run_id = 0  # Simulate the first run; change as needed
run_data = df[df["run_id"] == run_id].reset_index(drop=True)

# === Set up tracking ===
position = (1, 1)  # Starting position in simulated grid (can adjust)
path_trace = [position]

# Define movement directions
directions = {
    0: (0, -1),   # N
    1: (1, -1),   # NE
    2: (1, 0),    # E
    3: (1, 1),    # SE
    4: (0, 1),    # S
    5: (-1, 1),   # SW
    6: (-1, 0),   # W
    7: (-1, -1)   # NW
}

# === Simulate the robot step-by-step ===
for i in range(len(run_data)):
    row = run_data.iloc[i]
    features = pd.DataFrame([row[[f"sensor_{i}" for i in range(8)]]], columns=[f"sensor_{i}" for i in range(8)])
    predicted_action = model.predict(features)[0]

    dy, dx = directions.get(predicted_action, (0, 0))
    new_y = position[0] + dy
    new_x = position[1] + dx
    position = (new_y, new_x)
    path_trace.append(position)

# === Save the robot path ===
path_df = pd.DataFrame(path_trace, columns=["y", "x"])
path_df.to_csv("simulated_robot_path.csv", index=False)

print("✅ Simulation complete. Path saved to simulated_robot_path.csv")
