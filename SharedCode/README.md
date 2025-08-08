# Simulation Agent Training Pipeline

This project simulates a world with moving agents and uses collected data to train and evaluate a machine learning model.

## Project Structure

- `1_generate_data.py` — Generates synthetic sensor data from simulated world runs and saves it to a CSV.
- `2_train_models.py` — Loads the data from the CSV and trains a model.
- `3_run_simulation.py` — Uses the trained model to drive the agent in the simulated world and observe its behavior.
- `config.py` — Central config file where you define the training data filename and other global settings.
- `models/` — Stores the saved models and scalers.

---

## 🚀 How to Run

Make sure you have the necessary dependencies installed:

```bash
pip install xgboost

```

## Configuration

All settings (including which dataset to use) are defined in config.py.

```
WORLD_SIZE = 20
OBSTACLE_PROB = 0.1
WALL_COUNT = 0
WALL_MAX_LEN = 10
MIN_START_GOAL_DISTANCE = 8
SEED = 42
NUM_RUNS = 50000
ATTEMPTS = 100
TRAINING_DATA_FILE = "training_data_large_50k.csv"
```

```bash
python 1_generate_data
```

Filename schema:
Training data csv naming formats - `train_data_{sml|med|lrg}_{n}k.csv` where n is the abbreviated sample size in thousands

Experiment schema or naming conventions: - `{action|dir}_{norm}_{sml|med|lrg}_{sens8+dir}`
