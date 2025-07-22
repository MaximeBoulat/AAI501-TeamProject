---

# Robot Navigation Pathfinding and Dataset Generator 🧭🤖

This Python script simulates a robot navigating through randomly generated 2D grid worlds with obstacles, using the A\* pathfinding algorithm. It collects sensor-based training data for each step of the path and saves the results to a CSV file. This is ideal for building datasets for machine learning models that learn to navigate environments or mimic optimal paths.

---

## 📌 Features

* Generates multiple random 2D worlds filled with walls and obstacles.
* Uses the **A\*** search algorithm to find optimal paths from a random start to a random goal.
* Simulates **8-directional sensor readings** for each position along the path.
* Outputs a **training dataset** in CSV format, including timestamps, sensor data, chosen actions, and distance to goal.
* Saves a visualization of each world and its optimal path to disk.

---

## 📁 Files

* `robot_simulation.py`: Main simulation code.
* `robot_training_data.csv`: Output dataset (generated after running).
* `worlds/`: Folder where each world's `.npy` (binary) and `.txt` (human-readable) map files are saved.

---

## ▶️ How to Run

### Requirements

* Python 3.7+
* NumPy

Install required packages:

```bash
pip install numpy
```

Run the script:

```bash
python robot_simulation.py
```

---

## 🧠 How It Works — Step by Step

### 1. **World Creation**

Function: `create_world(...)`

* A 2D grid (`WORLD_SIZE` x `WORLD_SIZE`, default 20x20) is created.
* Each cell has a probability (`OBSTACLE_PROB`) of becoming an obstacle.
* Horizontal or vertical walls (up to `WALL_COUNT`, default 5) are added randomly to increase complexity.

### 2. **Start/Goal Selection**

Function: `simulate_world(...)`

* A random `start` and `goal` point are chosen in the world.
* The distance between them must exceed `MIN_START_GOAL_DISTANCE` to ensure interesting paths.
* If the A\* search can't find a path after `ATTEMPTS`, it retries with a new world.

### 3. **Pathfinding**

Function: `a_star(world, start, goal)`

* Uses the A\* algorithm with Euclidean distance as a heuristic.
* Finds the shortest path from `start` to `goal`, avoiding obstacles.

### 4. **Sensor Simulation**

Function: `get_sensor_readings(...)`

* At each step along the path, simulates **8-directional sensors**.
* Each sensor returns the number of free cells in its direction before hitting an obstacle or the edge of the world.

### 5. **Action Labeling**

Function: `get_action(from_pos, to_pos)`

* Converts a movement step into an action label (0 to 7), corresponding to one of the 8 directions.

### 6. **Training Data Generation**

Function: `generate_training_data(...)`

For each step:

* Records:

  * `timestamp` (global, sequentially increasing).
  * `run_id`
  * 8 sensor readings
  * action taken (0–7)
  * remaining distance to the goal

### 7. **Visualization**

Function: `print_world(...)`

* Prints the world map to the console with symbols:

  * `S` = start
  * `G` = goal
  * `#` = obstacle
  * `*` = path

### 8. **Saving Results**

* Worlds are saved to disk as:

  * `worlds/world_{run_id}.npy` — binary NumPy format
  * `worlds/world_{run_id}.txt` — text format for easy viewing
* All training data is saved to `robot_training_data.csv`

---

## 📊 CSV Output Format

File: `robot_training_data.csv`

| Column             | Description                                  |
| ------------------ | -------------------------------------------- |
| `timestamp`        | Global step index across all runs            |
| `run_id`           | ID of the simulation run                     |
| `sensor_0...7`     | Distance to obstacle in 8 directions         |
| `action`           | Direction taken (0–7)                        |
| `distance_to_goal` | Euclidean distance to goal from current cell |

---

## ⚙️ Configuration

You can modify these constants at the top of the script:

```python
num_runs = 50          # Number of simulation runs
WORLD_SIZE = 20        # Grid size (20x20)
ATTEMPTS = 100         # Max attempts to generate a valid world
WALL_COUNT = 5         # Number of random walls
WALL_MAX_LEN = 10      # Maximum wall length
OBSTALCE_PROB = 0.2    # Probability of single-tile obstacle
MIN_START_GOAL_DISTANCE = 8  # Minimum path length
SEED = 42              # Random seed for reproducibility
```

---

## ✅ Example Output

```text
=== World Map: Run 0 ===
. . . # . . . . . . . . . . . . . . . .
. # # # . . . . . . . . . . . . . . . .
. . . # . . . . . . . . . . . . . . . .
S * * * * * * * * * * * * * * * * * G .
. . . . . . . . . . . . . . . . . . . .
```

---

## 📈 Use Cases

* Dataset for training robot navigation models (RL or supervised).
* Preprocessing step for path-following ML tasks.
* Testbed for model evaluation or simulation-based planning.

---

## 🧪 Future Improvements

* Add GUI or animation for simulation playback.
* Support continuous sensor values (e.g., with Lidar simulation).
* Connect to RL frameworks like OpenAI Gym.

---
