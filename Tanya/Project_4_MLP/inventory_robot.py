# inventory_robot.py
import numpy as np
from sensors import read_sensors, DIRECTIONS

class InventoryRobot:
    def __init__(self, world, start, goal, model):
        self.world = world
        self.position = start
        self.goal = goal
        self.model = model
        self.path = [start]
        self.steps = 0

    def move(self, direction):
        dx, dy = direction
        new_x, new_y = self.position[0] + dx, self.position[1] + dy
        if (0 <= new_x < self.world.shape[0] and 0 <= new_y < self.world.shape[1] and self.world[new_x, new_y] != 1):
            self.position = (new_x, new_y)
            self.path.append(self.position)
            self.steps += 1
            return True
        return False

    def distance_to_goal(self, pos):
        return abs(pos[0] - self.goal[0]) + abs(pos[1] - self.goal[1])  # Manhattan

    def navigate(self, max_steps=100):
        while self.position != self.goal and self.steps < max_steps:
            features = read_sensors(self.world, self.position, self.goal)
            X_input = np.array(features).reshape(1, -1)
            prediction = self.model.predict(X_input)
            probs = self.model.predict_proba(X_input)
            best_dir_idx = prediction[0]
            confidence = np.max(probs)

            moved = False
            if confidence >= 0.10:
                moved = self.move(DIRECTIONS[best_dir_idx])

            if not moved:
                # Smart fallback
                fallback = []
                for i, d in enumerate(DIRECTIONS):
                    nx, ny = self.position[0] + d[0], self.position[1] + d[1]
                    if 0 <= nx < self.world.shape[0] and 0 <= ny < self.world.shape[1] and self.world[nx, ny] != 1:
                        dist = self.distance_to_goal((nx, ny))
                        fallback.append((dist, d))
                if fallback:
                    fallback.sort()
                    self.move(fallback[0][1])
                else:
                    print("Stuck with no fallback.")
                    return False
        return self.position == self.goal
