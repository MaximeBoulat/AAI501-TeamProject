import pandas as pd
import numpy as np
import random
from collections import defaultdict
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# Load data
df = pd.read_csv("robot_training_data.csv")

# Discretize sensor values into bins for Q-table compatibility
sensor_cols = [f"sensor_{i}" for i in range(8)]
n_bins = 5

est = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
df[sensor_cols] = est.fit_transform(df[sensor_cols])

# Train-test split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['action'])

# RL setup
actions = sorted(df["action"].unique())
action_to_idx = {a: i for i, a in enumerate(actions)}
idx_to_action = {i: a for a, i in action_to_idx.items()}
Q = defaultdict(lambda: np.zeros(len(actions)))

# Hyperparameters
alpha = 0.1      # learning rate
gamma = 0.9      # discount factor
epsilon = 0.1    # exploration rate
n_episodes = 10

# Q-learning training loop
for episode in range(n_episodes):
    print(f"training episode {episode} / {n_episodes}")
    for i in range(10000):
    #for i in range(len(train_df) - 1):
        state = tuple(train_df.iloc[i][sensor_cols].astype(int))
        action = train_df.iloc[i]["action"]
        reward = -train_df.iloc[i]["distance_to_goal"]

        next_state = tuple(train_df.iloc[i + 1][sensor_cols].astype(int))
        next_q = Q[next_state]

        a_idx = action_to_idx[action]
        best_next_action = np.argmax(next_q)
        td_target = reward + gamma * next_q[best_next_action]
        Q[state][a_idx] += alpha * (td_target - Q[state][a_idx])

# Inference function
def choose_action(sensor_input):
    state = tuple(est.transform([sensor_input])[0].astype(int))
    if random.random() < epsilon:
        return random.choice(actions)
    else:
        best_action_idx = np.argmax(Q[state])
        return idx_to_action[best_action_idx]

# Evaluate on test set
y_true = []
y_pred = []

for _, row in test_df.iterrows():
    sensors = row[sensor_cols].values
    true_action = row["action"]
    predicted_action = choose_action(sensors)
    y_true.append(true_action)
    y_pred.append(predicted_action)

# Compute accuracy and F1 score
accuracy = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average="macro")

print(accuracy)
print(f1)

