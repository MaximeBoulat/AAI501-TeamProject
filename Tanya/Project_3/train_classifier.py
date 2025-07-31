# train_classifier.py
import numpy as np
import os
import glob
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

X, y = [], []
for file in glob.glob("training_data/*.npz"):
    with np.load(file) as f:
        X.append(f['X'])
        y.append(f['y'])

if not X:
    raise RuntimeError("❌ No training data found in 'training_data/'.")

X = np.vstack(X)
y = np.hstack(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=300, max_depth=40, min_samples_split=2, random_state=42)
clf.fit(X_train, y_train)

print(f"✅ Training accuracy: {accuracy_score(y_train, clf.predict(X_train)):.2f}")
print(f"✅ Test accuracy: {accuracy_score(y_test, clf.predict(X_test)):.2f}")

joblib.dump(clf, 'robot_model.joblib')
