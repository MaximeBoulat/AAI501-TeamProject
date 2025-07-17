import pandas as pd
import numpy as np
import random
from collections import defaultdict
from sklearn.preprocessing import KBinsDiscretizer

# Load data
df = pd.read_csv("sensor_data.csv")

# Discretize sensor values into bins for Q-table compatibility
sensor_cols = [f"sensor_{i}" for i in range(8)]
n_bins = 5

est = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
df[sensor_cols] = est.fit_transform(df[sensor_cols])

# RL setup
actions = df["action"].unique()
state_size = n_bins ** len(sensor_cols)

# Q-table: state (tuple of sensor values) -> action -> Q-value
Q = defaultdict(lambda: np.zeros(len(actions)))

# Hyperparameters
alpha = 0.1      # learning rate
gamma = 0.9      # discount factor
epsilon = 0.1    # exploration rate
n_episodes = 50

# Convert action to index mapping
action_to_idx = {a: i for i, a in enumerate(sorted(actions))}
idx_to_action = {i: a for a, i in action_to_idx.items()}

# Training loop
for episode in range(n_episodes):
    for i in range(len(df) - 1):
        state = tuple(df.loc[i, sensor_cols].astype(int))
        action = df.loc[i, "action"]
        reward = -df.loc[i, "distance_to_goal"]  # Negative reward = better when closer

        next_state = tuple(df.loc[i + 1, sensor_cols].astype(int))
        next_q = Q[next_state]

        # Update Q-table using Q-learning rule
        a_idx = action_to_idx[action]
        best_next_action = np.argmax(next_q)
        td_target = reward + gamma * next_q[best_next_action]
        Q[state][a_idx] += alpha * (td_target - Q[state][a_idx])

# Inference example
def choose_action(sensor_input):
    state = tuple(est.transform([sensor_input])[0].astype(int))
    if random.random() < epsilon:
        return random.choice(actions)
    else:
        best_action_idx = np.argmax(Q[state])
        return idx_to_action[best_action_idx]

# Example usage
sample_input = [5, 2, 3, 4, 1, 3, 4, 2]
print("Chosen action for sensors", sample_input, ":", choose_action(sample_input))

