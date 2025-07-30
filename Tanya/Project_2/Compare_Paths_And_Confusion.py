import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# === Load data ===
astar_df = pd.read_csv("robot_data_with_astar_labels.csv")
ml_path_df = pd.read_csv("simulated_robot_path.csv")

# === 1. Compare A* path and ML path on a dual plot ===
def compare_paths():
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Simulated ML path
    axs[0].plot(ml_path_df['x'], ml_path_df['y'], marker='o', color='blue')
    axs[0].scatter(ml_path_df['x'].iloc[0], ml_path_df['y'].iloc[0], color='green', s=100, label='Start')
    axs[0].scatter(ml_path_df['x'].iloc[-1], ml_path_df['y'].iloc[-1], color='red', s=100, label='Goal')
    axs[0].invert_yaxis()
    axs[0].set_title("ML Predicted Path")
    axs[0].grid(True)
    axs[0].legend()

    # Use example A* path from index 0
    from generate_labels_with_astar import generate_grid_from_sensors, astar
    grid = generate_grid_from_sensors(astar_df.iloc[0])
    path = astar(grid, (1, 1), (0, 1))
    if path:
        y_coords, x_coords = zip(*path)
        axs[1].plot(x_coords, y_coords, marker='o', color='purple')
        axs[1].scatter(x_coords[0], y_coords[0], color='green', s=100, label='Start')
        axs[1].scatter(x_coords[-1], y_coords[-1], color='red', s=100, label='Goal')
        axs[1].invert_yaxis()
        axs[1].set_title("A* Path (Single Step)")
        axs[1].grid(True)
        axs[1].legend()

    plt.suptitle("ML vs A* Path Comparison")
    plt.tight_layout()
    plt.savefig("compare_paths.png")
    plt.show()

# === 2. Confusion Matrix: actual action vs A* action ===
def plot_confusion():
    y_true = astar_df['action']
    y_pred = astar_df['astar_action']
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues', xticks_rotation=45)
    plt.title("Confusion Matrix: Action vs A* Action")
    plt.tight_layout()
    plt.savefig("confusion_matrix_astar_vs_action.png")
    plt.show()

# === 3. Deep Dive: disagreement analysis ===
def analyze_disagreements():
    mismatches = astar_df[astar_df['action'] != astar_df['astar_action']].copy()
    total = len(astar_df)
    mismatch_pct = len(mismatches) / total * 100
    print(f"\n🔍 Disagreements between human label and A*: {len(mismatches)} out of {total} rows ({mismatch_pct:.2f}%)")

    print("\nExample mismatches (first 5 rows):")
    print(mismatches[['timestamp', 'run_id', 'action', 'astar_action', 'distance_to_goal'] + [f'sensor_{i}' for i in range(8)]].head())

    mismatches.to_csv("disagreement_log.csv", index=False)
    print("\n💾 Full disagreement log saved as disagreement_log.csv")

    # === Disagreements by action bar plot ===
    mismatch_counts = mismatches['action'].value_counts().sort_index()
    mismatch_counts.plot(kind='bar', color='tomato')
    plt.title("Disagreements by Human Action Label")
    plt.xlabel("Human Action")
    plt.ylabel("Disagreement Count")
    plt.tight_layout()
    plt.savefig("disagreements_by_action.png")
    plt.show()

    # === Sensor heatmap for mismatches ===
    sensor_cols = [f"sensor_{i}" for i in range(8)]
    avg = mismatches[sensor_cols].mean().to_frame().T
    sns.heatmap(avg, annot=True, cmap='Reds')
    plt.title("Avg Sensor Readings (Mismatches)")
    plt.tight_layout()
    plt.savefig("sensor_heatmap_mismatches.png")
    plt.show()

    # === Sorted by distance to goal ===
    print("\nTop 5 disagreements with longest distance to goal:")
    print(mismatches.sort_values(by='distance_to_goal', ascending=False).head())

    # === Disagreements clustered by run_id ===
    run_counts = mismatches['run_id'].value_counts().sort_index()
    run_counts.plot(kind='bar', color='orange')
    plt.title("Disagreements by Run ID")
    plt.xlabel("Run ID")
    plt.ylabel("Disagreement Count")
    plt.tight_layout()
    plt.savefig("disagreements_by_runid.png")
    plt.show()

# === Run all analyses ===
compare_paths()
plot_confusion()
analyze_disagreements()