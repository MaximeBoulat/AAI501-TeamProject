# simulate_robot.py
import numpy as np
import random
import time
from sensors import read_sensors, DIRECTIONS
from robot_agent import InventoryRobot, print_grid  # you'll add this helper from earlier
from train import clf  # your trained model

def get_random_position(world):
    free = np.argwhere(world == 0)
    return tuple(random.choice(free))

world = np.load("../data/worlds/world_0.npy")

# Define start and goal
start = get_random_position(world)
goal = get_random_position(world)
while goal == start:
    goal = get_random_position(world)

# Mark them on the world copy (purely visual)
world_visual = world.copy()
world_visual[start] = 2  # Start
world_visual[goal] = 3   # Goal

robot = InventoryRobot(world, start, goal, clf)
robot.navigate(max_steps=100)

def print_grid(grid, robot_pos, goal_pos):
    os.system("cls" if os.name == "nt" else "clear")
    for i in range(grid.shape[0]):
        row = ""
        for j in range(grid.shape[1]):
            if (i, j) == robot_pos:
                row += " R "
            elif (i, j) == goal_pos:
                row += " G "
            elif grid[i, j] == 1:
                row += "███"
            else:
                row += " . "
        print(row)
    time.sleep(0.1)
