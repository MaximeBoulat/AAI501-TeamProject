import os
import numpy as np
import joblib
from sklearn.neural_network import MLPClassifier
from sensors import read_sensors

world_dir = "worlds"
X, y = [], []

for file in os.listdir(world_dir):
    if file.endswith(".npy"):
        world = np.load(os.path.join(world_dir, file))
        goal = tuple(np.argwhere(world == 3)[0])
        start = tuple(np.argwhere(world == 2)[0])
        pos = start
        for _ in range(30):
            features = read_sensors(world, pos, goal)
            for idx, direction in enumerate([(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]):
                X.append(features)
                y.append(idx)
clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300)
clf.fit(X, y)
joblib.dump(clf, "robot_model_mlp.joblib")
print("✅ MLP Neural Network model trained and saved.")