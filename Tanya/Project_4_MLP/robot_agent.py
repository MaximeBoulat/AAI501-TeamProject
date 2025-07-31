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
        new_pos = (new_x, new_y)

        if (0 <= new_x < self.world.shape[0] and
            0 <= new_y < self.world.shape[1] and
            self.world[new_pos] != 1):
            self.position = new_pos
            self.path.append(new_pos)
            self.steps += 1
            return True
        return False

    def distance_to_goal(self, pos):
        return abs(pos[0] - self.goal[0]) + abs(pos[1] - self.goal[1])  # Manhattan distance

    def navigate(self, max_steps=50):
        while self.position != self.goal and self.steps < max_steps:
            print(f"\n--- STEP {self.steps + 1} ---")
            print(f"Current position: {self.position}")
            print(f"Goal position: {self.goal}")

            features = read_sensors(self.world, self.position, self.goal)  # ✅ FIXED HERE
            X_input = np.array(features).reshape(1, -1)

            prediction = self.model.predict(X_input)
            probs = self.model.predict_proba(X_input)
            max_confidence = np.max(probs)
            best_direction = DIRECTIONS[prediction[0]]

            moved = False
            if max_confidence >= 0.10:
                moved = self.move(best_direction)
                if moved:
                    print(f"✅ Moving {best_direction} with confidence {max_confidence:.2f}")

            if not moved:
                print("⚠️  No confident prediction. Trying smarter fallback...")
                fallback_moves = []
                for d in DIRECTIONS:
                    nx, ny = self.position[0] + d[0], self.position[1] + d[1]
                    next_pos = (nx, ny)
                    if (0 <= nx < self.world.shape[0] and
                        0 <= ny < self.world.shape[1] and
                        self.world[next_pos] != 1):
                        dist = self.distance_to_goal(next_pos)
                        fallback_moves.append((dist, d))

                if fallback_moves:
                    fallback_moves.sort()
                    best_fallback = fallback_moves[0][1]
                    self.move(best_fallback)
                    print(f"🚶 Smarter fallback moving {best_fallback}")
                else:
                    print("❌ Robot stuck. No valid fallback.")
                    print("\n⚠️ Final path before giving up:")
                    print(self.path)
                    return False

        if self.position == self.goal:
            print("\n🎯 Goal reached!")
        else:
            print("\n⚠️ Max steps reached before goal.")
        print(f"\n✅ Final path taken:\n{self.path}")
        return self.position == self.goal



