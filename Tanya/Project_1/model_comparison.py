
# model_comparison.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# Load dataset
df = pd.read_csv("robot_training_data.csv")
sensor_cols = [f"sensor_{i}" for i in range(8)]
X = df[sensor_cols]
y = df["action"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Output setup
output_dir = "model_comparison_outputs"
os.makedirs(output_dir, exist_ok=True)

# ---------- Random Forest ----------
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred, average='weighted')
cm_rf = confusion_matrix(y_test, rf_pred)
ConfusionMatrixDisplay(confusion_matrix=cm_rf).plot(cmap="Blues")
plt.title("Random Forest Confusion Matrix")
plt.savefig(f"{output_dir}/confusion_rf.png")
plt.close()

# ---------- Naive Bayes ----------
nb = GaussianNB()
nb.fit(X_train, y_train)
nb_pred = nb.predict(X_test)
nb_acc = accuracy_score(y_test, nb_pred)
nb_f1 = f1_score(y_test, nb_pred, average='weighted')
cm_nb = confusion_matrix(y_test, nb_pred)
ConfusionMatrixDisplay(confusion_matrix=cm_nb).plot(cmap="Purples")
plt.title("Naive Bayes Confusion Matrix")
plt.savefig(f"{output_dir}/confusion_nb.png")
plt.close()

# ---------- Neural Network ----------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model = Sequential([
    Dense(64, activation='relu', input_shape=(8,)),
    Dense(32, activation='relu'),
    Dense(8, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train_scaled, y_train, validation_data=(X_test_scaled, y_test), epochs=20, batch_size=32, verbose=0)
nn_pred = np.argmax(model.predict(X_test_scaled), axis=1)
nn_acc = accuracy_score(y_test, nn_pred)
nn_f1 = f1_score(y_test, nn_pred, average='weighted')
cm_nn = confusion_matrix(y_test, nn_pred)
ConfusionMatrixDisplay(confusion_matrix=cm_nn).plot(cmap="Greens")
plt.title("Neural Network Confusion Matrix")
plt.savefig(f"{output_dir}/confusion_nn.png")
plt.close()

# ---------- Q-Learning (Assumed values) ----------
ql_acc = 0.2873
ql_f1 = 0.2520

# ---------- Summary DataFrame ----------
summary_df = pd.DataFrame({
    "Model": ["Neural Network", "Naive Bayes", "Random Forest", "Q-Learning"],
    "Accuracy": [nn_acc, nb_acc, rf_acc, ql_acc],
    "F1-Score": [nn_f1, nb_f1, rf_f1, ql_f1]
}).sort_values("Accuracy", ascending=False)

summary_df.to_csv(f"{output_dir}/model_comparison_summary.csv", index=False)

# Accuracy bar chart
plt.figure(figsize=(8, 5))
sns.barplot(x="Model", y="Accuracy", data=summary_df, hue="Model", legend=False, palette="Blues_d")
plt.title("Model Accuracy Comparison")
plt.ylim(0, 1)
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(f"{output_dir}/model_accuracy_comparison.png")
plt.close()

# F1 bar chart
plt.figure(figsize=(8, 5))
sns.barplot(x="Model", y="F1-Score", data=summary_df, hue="Model", legend=False, palette="Oranges")
plt.title("Model F1-Score Comparison")
plt.ylim(0, 1)
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(f"{output_dir}/model_f1_comparison.png")
plt.close()

print(f"✅ Model comparison complete. Outputs saved to '{output_dir}/'")
