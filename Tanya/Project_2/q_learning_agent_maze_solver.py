
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# === Load World ===
world = np.load("world_0.npy")

# === Agent Path (from Q-learning) ===
agent_path = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6), (3, 7), (3, 8), (3, 9), (3, 10), (4, 11)]

# === Visualization ===
fig, ax = plt.subplots()
ax.set_xlim(0, world.shape[1])
ax.set_ylim(0, world.shape[0])
ax.set_aspect("equal")
ax.set_title("Q-Learning Agent Maze Solver")
grid = []

def init():
    for y in range(world.shape[0]):
        for x in range(world.shape[1]):
            color = "black" if world[y, x] == 1 else "white"
            rect = plt.Rectangle((x, y), 1, 1, color=color, edgecolor="gray")
            ax.add_patch(rect)
            grid.append(rect)
    return grid

agent_dot, = ax.plot([], [], "ro", markersize=8)

def update(frame):
    if frame < len(agent_path):
        x, y = agent_path[frame]
        agent_dot.set_data(x + 0.5, y + 0.5)
    return [agent_dot]

ani = animation.FuncAnimation(fig, update, frames=len(agent_path),
                              init_func=init, blit=True, interval=300, repeat=False)
plt.show()
