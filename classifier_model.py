
# classifier_model.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Load dataset
df = pd.read_csv("robot_training_data.csv")
sensor_cols = [f"sensor_{i}" for i in range(8)]
X = df[sensor_cols]
y = df["action"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Create output directory
output_dir = "classifier_outputs"
os.makedirs(output_dir, exist_ok=True)

# Save accuracy
with open(f"{output_dir}/accuracy.txt", "w") as f:
    f.write(f"Random Forest Accuracy: {accuracy:.4f}\n")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues", xticks_rotation=45)
plt.title("Confusion Matrix - Random Forest")
plt.savefig(f"{output_dir}/confusion_matrix.png")
plt.close()

# Feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x=clf.feature_importances_, y=sensor_cols)
plt.title("Feature Importance - Random Forest")
plt.xlabel("Importance")
plt.ylabel("Sensor")
plt.tight_layout()
plt.savefig(f"{output_dir}/feature_importance.png")
plt.close()

print(f"✅ Classifier complete. Accuracy: {accuracy:.4f}")
print(f"Outputs saved to: {output_dir}/")
