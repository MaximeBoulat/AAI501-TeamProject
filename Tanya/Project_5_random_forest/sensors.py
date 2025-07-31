# sensors.py (Updated + Includes get_direction_label)
import numpy as np

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
            features.append(1)

    # 2. Relative direction to goal
    dx = gx - x
    dy = gy - y
    features.extend([np.sign(dx), np.sign(dy)])

    # 3. Normalized position
    features.append(x / (rows - 1))
    features.append(y / (cols - 1))

    # 4. 3x3 flattened local grid
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols:
                features.append(grid[nx, ny])
            else:
                features.append(1)

    # 5. One-hot encoding of local position
    one_hot = np.zeros(9)
    one_hot[((x % 3) * 3 + (y % 3)) % 9] = 1
    features.extend(one_hot.tolist())

    # 6. Goal angle
    angle = np.arctan2(gy - y, gx - x)
    features.append(angle / np.pi)

    # 7. Goal quadrant
    quadrant = ((np.around(np.degrees(angle)) + 360) % 360) // 45
    features.append(int(quadrant))

    # 8. Normalized distances
    manhattan = (abs(dx) + abs(dy)) / (rows + cols)
    euclidean = (np.sqrt(dx ** 2 + dy ** 2)) / np.sqrt(rows**2 + cols**2)
    features.append(manhattan)
    features.append(euclidean)

    assert len(features) == 34, f"Expected 34 features, got {len(features)}"
    return np.array(features)

# ✅ Add this function so import works
def get_direction_label(current_pos, next_pos):
    dx = next_pos[0] - current_pos[0]
    dy = next_pos[1] - current_pos[1]
    return (dx, dy)

def get_direction_label(current, next_pos):
    dx = next_pos[0] - current[0]
    dy = next_pos[1] - current[1]

    if dx == -1 and dy == 0:
        return 0  # Up
    elif dx == 1 and dy == 0:
        return 1  # Down
    elif dx == 0 and dy == -1:
        return 2  # Left
    elif dx == 0 and dy == 1:
        return 3  # Right
    elif dx == -1 and dy == -1:
        return 4  # Up-Left
    elif dx == -1 and dy == 1:
        return 5  # Up-Right
    elif dx == 1 and dy == -1:
        return 6  # Down-Left
    elif dx == 1 and dy == 1:
        return 7  # Down-Right
    else:
        raise ValueError(f"Invalid move from {current} to {next_pos}")






  
