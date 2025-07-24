# eda_analysis.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load dataset
df = pd.read_csv("robot_training_data.csv")

# Define sensor column names
sensor_cols = [f"sensor_{i}" for i in range(8)]

# Create output folder
output_dir = "eda_outputs"
os.makedirs(output_dir, exist_ok=True)

# Sensor Value Distributions
plt.figure(figsize=(12, 6))
df[sensor_cols].boxplot()
plt.title("Sensor Value Distributions (All 8 Directions)")
plt.ylabel("Distance to Obstacle")
plt.savefig(f"{output_dir}/sensor_boxplots.png")
plt.close()

# Action Frequency Plot
plt.figure(figsize=(8, 5))
sns.countplot(x="action", data=df)
plt.title("Action Frequency Distribution")
plt.xlabel("Action (Direction 0-7)")
plt.ylabel("Count")
plt.savefig(f"{output_dir}/action_distribution.png")
plt.close()

# Correlation Matrix (Sensors + Distance to Goal)
corr = df[sensor_cols + ["distance_to_goal"]].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix (Sensors & Distance to Goal)")
plt.savefig(f"{output_dir}/correlation_matrix.png")
plt.close()

# Sensor Readings by Action (Grouped Boxplots)
plt.figure(figsize=(14, 8))
for i, col in enumerate(sensor_cols):
    plt.subplot(2, 4, i + 1)
    sns.boxplot(x="action", y=col, data=df)
    plt.title(f"{col} by Action")
plt.tight_layout()
plt.savefig(f"{output_dir}/sensors_by_action.png")
plt.close()

# Distance to Goal Over Time
plt.figure(figsize=(12, 4))
sns.lineplot(x="timestamp", y="distance_to_goal", data=df)
plt.title("Distance to Goal Over Time")
plt.xlabel("Timestamp")
plt.ylabel("Distance to Goal")
plt.savefig(f"{output_dir}/distance_over_time.png")
plt.close()

# Run-Level Summary Statistics
summary = df.groupby("run_id")["distance_to_goal"].agg(["mean", "min", "max", "std"])
summary.to_csv(f"{output_dir}/run_summary_stats.csv")

# Optional: Distance to Goal Boxplot by Run
plt.figure(figsize=(12, 4))
sns.boxplot(x="run_id", y="distance_to_goal", data=df)
plt.title("Distance to Goal by Run ID")
plt.xticks(rotation=90)
plt.savefig(f"{output_dir}/distance_by_run.png")
plt.close()

print("✅ EDA complete. All plots and summary saved to 'eda_outputs/' folder.")
