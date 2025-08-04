# updated_eda_analysis.py
#
# This script performs exploratory data analysis (EDA) on the robot navigation
# dataset that includes eight sensor readings, the Euclidean distance to the
# goal, and the normalized goal direction.  It generates several plots to
# visualize feature distributions and correlations and writes a simple
# summary of contradictory sensor patterns.  The script assumes that the
# dataset is stored as a CSV file named ``training_data.csv`` in the same
# directory.

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def run_eda(csv_file: str = "training_data.csv", output_dir: str = "eda_outputs_updated") -> None:
    """Perform exploratory data analysis on the navigation dataset.

    Args:
        csv_file: Path to the CSV file containing training data.
        output_dir: Directory where plots and summaries will be saved.

    The function generates the following outputs:
        - sensor_boxplots_by_action.png: Boxplots of each sensor by action.
        - correlation_matrix_updated.png: Heatmap of feature correlations.
        - goal_direction_distribution.png: Histogram of normalized goal direction.
        - action_distribution.png: Bar chart of action frequencies.
        - contradiction_summary.txt: Text file reporting the number of sensor patterns
          that map to more than one action (shifting‑signals metric).
    """
    df = pd.read_csv(csv_file)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    sensor_cols = [f"sensor_{i}" for i in range(8)]

    # Boxplots of sensors by action
    plt.figure(figsize=(14, 8))
    for i, col in enumerate(sensor_cols):
        plt.subplot(2, 4, i + 1)
        sns.boxplot(x="action", y=col, data=df)
        plt.title(f"{col} by Action")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "sensor_boxplots_by_action.png"))
    plt.close()

    # Correlation matrix including distance and goal_direction
    corr = df[sensor_cols + ["distance_to_goal", "goal_direction"]].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title("Correlation Matrix with Goal Direction")
    plt.savefig(os.path.join(output_dir, "correlation_matrix_updated.png"))
    plt.close()

    # Distribution of goal_direction
    plt.figure(figsize=(8, 4))
    sns.histplot(df["goal_direction"], bins=30, kde=True)
    plt.title("Distribution of Normalized Goal Direction")
    plt.xlabel("Goal Direction (normalized)")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(output_dir, "goal_direction_distribution.png"))
    plt.close()

    # Action distribution
    plt.figure(figsize=(8, 4))
    sns.countplot(x="action", data=df)
    plt.title("Action Frequency Distribution")
    plt.xlabel("Action (0–7)")
    plt.ylabel("Count")
    plt.savefig(os.path.join(output_dir, "action_distribution.png"))
    plt.close()

    # Compute shifting‑signals summary: number of sensor patterns with multiple actions
    sensor_only_group = df.groupby(sensor_cols)["action"].nunique()
    contradictory_patterns = (sensor_only_group > 1).sum()
    total_patterns = len(sensor_only_group)
    proportion = contradictory_patterns / total_patterns
    summary_path = os.path.join(output_dir, "contradiction_summary.txt")
    with open(summary_path, "w") as f:
        f.write(
            f"Contradictory sensor patterns: {contradictory_patterns} out of {total_patterns} "
            f"({proportion:.3%} of unique patterns)\n"
        )

    print(f"EDA complete. Outputs saved to {output_dir}/")


if __name__ == "__main__":
    run_eda()