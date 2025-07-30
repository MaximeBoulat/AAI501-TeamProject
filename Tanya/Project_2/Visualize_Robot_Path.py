# Visualize Robot Path
import pandas as pd
import matplotlib.pyplot as plt

# Load the simulated path
path_df = pd.read_csv("simulated_robot_path.csv")

# Create a scatter plot with lines
plt.figure(figsize=(8, 8))
plt.plot(path_df['x'], path_df['y'], marker='o', linestyle='-', color='blue')

# Mark start and end points
plt.scatter(path_df['x'].iloc[0], path_df['y'].iloc[0], color='green', s=100, label='Start')
plt.scatter(path_df['x'].iloc[-1], path_df['y'].iloc[-1], color='red', s=100, label='Goal')

# Customize plot
plt.title("Simulated Robot Path")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.grid(True)
plt.legend()
plt.gca().invert_yaxis()  # Invert y-axis to match typical grid layout
plt.tight_layout()
plt.savefig("robot_path_plot.png")
plt.show()
