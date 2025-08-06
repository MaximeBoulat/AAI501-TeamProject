"""
Expanded Exploratory Data Analysis for the robot navigation dataset.

This script loads the updated training dataset, computes several
diagnostic plots and metrics to better understand the feature
distributions and the limitations of the supervised navigation
problem.  The outputs include:

1. A correlation matrix heatmap for all sensor features plus
   distance_to_goal and goal_direction.
2. Feature importances derived from a Random Forest classifier,
   illustrating which inputs are most predictive of the expert action.
3. A histogram quantifying the "shifting signals" problem by
   counting how many distinct actions map to identical (binned)
   sensor configurations.
4. A heatmap showing the conditional probability of each action
   given the binned goal direction, highlighting how the expert
   action depends on the relative orientation to the goal.

The generated plots are saved to the directory `new_eda_outputs` in
the current working directory.  To run this script, ensure that
`training_data.csv` is present in the working directory.  You may
need to install pandas, seaborn, matplotlib and scikit-learn.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier


def main():
    # Load the dataset
    df = pd.read_csv("training_data.csv")

    # Ensure output directory exists
    out_dir = "new_eda_outputs"
    os.makedirs(out_dir, exist_ok=True)

    # Select features and target
    sensor_cols = [f"sensor_{i}" for i in range(8)]
    feature_cols = sensor_cols + ["distance_to_goal", "goal_direction"]
    X = df[feature_cols]
    y = df["action"]

    # 1. Correlation matrix
    corr = X.corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Correlation matrix (updated dataset)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "corr_updated.png"))
    plt.close()

    # 2. Feature importances via Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(8, 4))
    plt.bar(range(len(importances)), importances[indices], align="center")
    plt.xticks(range(len(importances)), [feature_cols[i] for i in indices], rotation=90)
    plt.title("Random Forest Feature Importances")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "feature_importance.png"))
    plt.close()

    # 3. Contradiction analysis
    # Bin sensor values into three categories: low (<5), medium (5–10), high (>10)
    bins = [0, 5, 10, np.inf]
    binned = df[sensor_cols].apply(lambda col: pd.cut(col, bins, labels=False, include_lowest=True))
    # Group by the binned sensor pattern and count unique actions
    # We use tuple of binned values as the group key
    pattern_actions = binned.copy()
    pattern_actions["action"] = y
    contradictions = pattern_actions.groupby(sensor_cols)["action"].nunique()
    # Histogram of unique action counts
    count_vals = contradictions.value_counts().sort_index()
    plt.figure()
    count_vals.plot(kind="bar")
    plt.xlabel("Number of distinct actions per binned sensor pattern")
    plt.ylabel("Count of sensor patterns")
    plt.title("Contradictory labeling analysis")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "contradiction_hist.png"))
    plt.close()

    # 4. Action probabilities conditioned on goal direction
    # Discretize goal_direction into 8 bins (0–1 scale)
    bins_dir = np.linspace(0, 1, 9)
    labels = [f"{i/8:.2f}-{(i+1)/8:.2f}" for i in range(8)]
    df["dir_bin"] = pd.cut(df["goal_direction"], bins_dir, labels=labels, include_lowest=True)
    action_counts = df.groupby(["dir_bin", "action"]).size().unstack(fill_value=0)
    # Normalize rows to obtain probabilities
    prob = action_counts.div(action_counts.sum(axis=1), axis=0)
    plt.figure(figsize=(8, 6))
    sns.heatmap(prob, annot=False, cmap="viridis")
    plt.title("Action probabilities by goal direction bin")
    plt.xlabel("Action")
    plt.ylabel("Goal direction bin")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "dir_action_heatmap.png"))
    plt.close()

    print(f"EDA complete. Plots saved to '{out_dir}' directory.")


if __name__ == "__main__":
    main()