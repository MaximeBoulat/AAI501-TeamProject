
# bayesian_model.py
import pandas as pd
import os
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("robot_training_data.csv")
sen

sor_cols = [f"sensor_{i}" for i in range(8)]
X = df[sensor_cols]
y = df["action"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Naive Bayes model
model = GaussianNB()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Output directory
output_dir = "bayesian_outputs"
os.makedirs(output_dir, exist_ok=True)

# Save accuracy
with open(f"{output_dir}/accuracy.txt", "w") as f:
    f.write(f"Naive Bayes Accuracy: {accuracy:.4f}\n")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Purples", xticks_rotation=45)
plt.title("Confusion Matrix - Naive Bayes")
plt.savefig(f"{output_dir}/confusion_matrix.png")
plt.close()

print(f"✅ Naive Bayes model complete. Accuracy: {accuracy:.4f}")
print(f"Outputs saved to: {output_dir}/")
