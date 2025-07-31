# sensors.py (Version 2 – Upgraded)
import numpy as np

# Directions used for prediction mapping
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1),  # Up, Down, Left, Right
              (-1, -1), (-1, 1), (1, -1), (1, 1)]  # Diagonals

def read_sensors(grid, pos, goal):
    x, y = pos
    gx, gy = goal
    rows, cols = grid.shape

    features = []

    # 1. 8 surrounding tiles
    for dx, dy in DIRECTIONS:
        nx, ny = x + dx, y + dy
        if 0 <= nx < rows and 0 <= ny < cols:
            features.append(grid[nx, ny])
        else:
            features.append(1)  # wall/out-of-bounds

    # 2. Relative direction to goal (dx, dy)
    dx = gx - x
    dy = gy - y
    norm_dx = np.sign(dx)
    norm_dy = np.sign(dy)
    features.extend([norm_dx, norm_dy])

    # 3. Normalized position of agent (x, y)
    features.append(x / (rows - 1))
    features.append(y / (cols - 1))

    # 4. 3x3 flattened local grid
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols:
                features.append(grid[nx, ny])
            else:
                features.append(1)  # out of bounds

    # 5. One-hot encoding of 3x3 agent position (9)
    one_hot = np.zeros(9)
    one_hot[((x % 3) * 3 + (y % 3)) % 9] = 1
    features.extend(one_hot.tolist())

    # --- New Additions ---
    # 6. Goal angle (1)
    angle = np.arctan2(gy - y, gx - x)  # atan2(dy, dx)
    features.append(angle / np.pi)  # Normalize angle to [-1, 1]

    # 7. Goal quadrant (1) from agent perspective
    quadrant = ((np.around(np.degrees(angle)) + 360) % 360) // 45
    features.append(int(quadrant))

    # 8. Normalized Manhattan + Euclidean distance to goal (2)
    manhattan = (abs(dx) + abs(dy)) / (rows + cols)
    euclidean = (np.sqrt(dx ** 2 + dy ** 2)) / np.sqrt(rows**2 + cols**2)
    features.append(manhattan)
    features.append(euclidean)

    assert len(features) == 34, f"Expected 34 features, got {len(features)}"
    return np.array(features)






  
