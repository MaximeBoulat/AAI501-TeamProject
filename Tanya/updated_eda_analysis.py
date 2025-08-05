"""
updated_eda_analysis.py
----------------------

This script performs exploratory data analysis (EDA) on the navigation training
dataset produced by the simulator.  It assumes that the dataset contains eight
sensor readings (`sensor_0`–`sensor_7`), the Euclidean distance to the goal
(`distance_to_goal`), the normalized goal direction (`goal_direction`), and
the discrete action taken (`action`).

The script computes summary statistics, generates diagnostic plots, and
quantifies the frequency of contradictory labels for identical sensor
configurations.  Plots are saved into an `updated_eda_outputs` directory.

Usage:
    python updated_eda_analysis.py [path/to/training_data.csv]

If no path is provided, the script defaults to `training_data.csv` in the
current working directory.
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def perform_eda(csv_path: str = "training_data.csv") -> None:
    """Load the dataset from `csv_path` and perform EDA, saving outputs."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find training data file: {csv_path}")

    # Load data
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} records from {csv_path}")

    # Create output directory
    output_dir = "updated_eda_outputs"
    os.makedirs(output_dir, exist_ok=True)

    # Define feature columns
    sensor_cols = [f"sensor_{i}" for i in range(8)]
    feature_cols = sensor_cols + ["distance_to_goal", "goal_direction"]

    # Summary statistics
    summary = df[feature_cols + ["action"]].describe().T
    summary_path = os.path.join(output_dir, "summary_statistics.csv")
    summary.to_csv(summary_path)
    print(f"Summary statistics saved to {summary_path}")

    # Correlation matrix
    corr = df[feature_cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Correlation Matrix (Sensors, Distance, Goal Direction)")
    plt.tight_layout()
    corr_path = os.path.join(output_dir, "correlation_matrix_updated.png")
    plt.savefig(corr_path)
    plt.close()
    print(f"Correlation matrix saved to {corr_path}")

    # Boxplots of sensor readings by action
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(12, 16))
    for i, ax in enumerate(axes.flatten()):
        col = sensor_cols[i]
        sns.boxplot(x="action", y=col, data=df, ax=ax)
        ax.set_title(f"Sensor {i} by Action")
        ax.set_xlabel("Action")
        ax.set_ylabel("Distance to obstacle")
    fig.suptitle("Sensor readings distribution by action", y=1.02)
    plt.tight_layout()
    boxplot_path = os.path.join(output_dir, "sensor_boxplots_by_action.png")
    fig.savefig(boxplot_path)
    plt.close(fig)
    print(f"Sensor boxplots saved to {boxplot_path}")

    # Goal direction distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(df["goal_direction"], bins=20, kde=True, color="purple")
    plt.xlabel("Normalized goal direction")
    plt.ylabel("Frequency")
    plt.title("Distribution of Goal Direction")
    plt.tight_layout()
    goal_dist_path = os.path.join(output_dir, "goal_direction_distribution.png")
    plt.savefig(goal_dist_path)
    plt.close()
    print(f"Goal direction distribution saved to {goal_dist_path}")

    # Action distribution
    plt.figure(figsize=(8, 5))
    sns.countplot(x="action", data=df)
    plt.title("Distribution of Actions")
    plt.xlabel("Action (0–7)")
    plt.ylabel("Count")
    plt.tight_layout()
    action_dist_path = os.path.join(output_dir, "action_distribution.png")
    plt.savefig(action_dist_path)
    plt.close()
    print(f"Action distribution saved to {action_dist_path}")

    # Contradiction analysis: number of unique actions per sensor pattern
    pattern_group = df.groupby(sensor_cols)["action"].nunique()
    contradiction_counts = pattern_group.value_counts().sort_index()
    # Save summary to text file
    contradiction_path = os.path.join(output_dir, "contradiction_summary.txt")
    with open(contradiction_path, "w") as f:
        f.write("Unique optimal actions per sensor pattern:\n")
        for num_actions, count in contradiction_counts.items():
            f.write(f"{num_actions} unique action(s): {count} patterns\n")
    print(f"Contradiction summary saved to {contradiction_path}")


if __name__ == "__main__":
    # Accept optional command-line argument for CSV path
    csv_arg = sys.argv[1] if len(sys.argv) > 1 else "training_data.csv"
    perform_eda(csv_arg)