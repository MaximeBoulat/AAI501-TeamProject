
# neural_net_model.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import plot_model

# Load dataset
df = pd.read_csv("robot_training_data.csv")
sensor_cols = [f"sensor_{i}" for i in range(8)]
X = df[sensor_cols]
y = df["action"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build neural network
model = Sequential([
    Dense(64, activation='relu', input_shape=(8,)),
    Dense(32, activation='relu'),
    Dense(8, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train
history = model.fit(X_train_scaled, y_train, validation_data=(X_test_scaled, y_test), epochs=20, batch_size=32, verbose=0)

# Output folder
output_dir = "neural_net_outputs"
os.makedirs(output_dir, exist_ok=True)

# Accuracy plot
plt.figure()
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Neural Network Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(f"{output_dir}/accuracy_plot.png")
plt.close()

# Save final accuracy
final_acc = history.history['val_accuracy'][-1]
with open(f"{output_dir}/accuracy.txt", "w") as f:
    f.write(f"Final Validation Accuracy: {final_acc:.4f}\n")

print(f"✅ Neural network complete. Final val accuracy: {final_acc:.4f}")
print(f"Outputs saved to: {output_dir}/")
