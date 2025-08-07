
WORLD_SIZE = 20
OBSTACLE_PROB = 0.1
WALL_COUNT = 0
WALL_MAX_LEN = 10
MIN_START_GOAL_DISTANCE = 8
SEED = 42
ATTEMPTS = 100
X_COLS=[f'sensor_{i}' for i in range(8)] + ['distance_to_goal', 'goal_direction']
Y_HAT_COL=['action']
NUM_RUNS = 10000

# Dynamically Generate Required names based on the above vars 
size_label = "small"
if NUM_RUNS > 3000:
    size_label = "medium"
elif NUM_RUNS > 10000:
    size_label = "large"

def condense(columns: list[str]) -> str:
    summary = []
    if any(col.startswith('sensor_') for col in columns):
        summary.append('sensor')
    if 'goal_direction' in columns: 
        summary.append('dist')
    if 'distance_to_goal' in columns: 
        summary.append('dir')
    return '+'.join(summary)

EXPERIMENT_NAME=f"{Y_HAT_COL[0]}_norm_{size_label}_{condense(X_COLS)}"
TRAINING_DATA_FILE=f"training_data_{size_label}_{NUM_RUNS // 1_000}k.csv"
