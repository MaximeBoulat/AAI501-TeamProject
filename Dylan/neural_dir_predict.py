# neural_net_direction_model.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import plot_model
import numpy as np

# Load dataset
df = pd.read_csv("robot_training_data.csv")

sensor_cols = [f"sensor_{i}" for i in range(8)]
X = df[sensor_cols]
y_angle = df["goal_direction"].astype(float)

# Normalize angle (in radians) to [0, 1] for regression
y_normalized = y_angle / (2 * np.pi)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_normalized, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build regression model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')  # output normalized angle
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train
history = model.fit(X_train_scaled, y_train, validation_data=(X_test_scaled, y_test), epochs=30, batch_size=32, verbose=0)

# Output directory
output_dir = "neural_net_outputs"
os.makedirs(output_dir, exist_ok=True)

# Loss plot
plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Direction-to-Goal Regression Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.savefig(f"{output_dir}/loss_plot.png")
plt.close()

# Save final MAE
final_mae = history.history['val_mae'][-1]
with open(f"{output_dir}/mae.txt", "w") as f:
    f.write(f"Final Validation MAE: {final_mae:.4f}\n")

print(f"✅ Regression complete. Final val MAE: {final_mae:.4f}")
print(f"Outputs saved to: {output_dir}/")

