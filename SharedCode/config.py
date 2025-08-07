
WORLD_SIZE = 20
OBSTACLE_PROB = 0.1
WALL_COUNT = 0
WALL_MAX_LEN = 10
MIN_START_GOAL_DISTANCE = 8
SEED = 42
NUM_RUNS = 3000
ATTEMPTS = 100
size_label = "small"
if NUM_RUNS > 3000:
    size_label = "medium"
elif NUM_RUNS > 10000:
    size_label = "large"
TRAINING_DATA_FILE=f"training_data_{size_label}_{NUM_RUNS // 1_000}k.csv"

# What the experiment should be called so we can track this version in understandable words 
# rather than just a number
#X_INPUT_COLUMN_NAMES=
#Y_YHAT_COLUMN_NAMES=
EXPERIMENT_NAME=f"{Y_YHAT_COLUMN_NAMES}_norm_{size_label}_{X_INPUT_COLUMN_NAMES.contains(sensor)+X_INPUT_COLUMN_NAMES.contains(dir)}"
