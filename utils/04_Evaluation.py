# ====================================
# 04_Evaluation.py
# Model Evaluation Visuals
# ====================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import sys
import os
import platform
import subprocess
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load model
model = joblib.load('models/netflix_popularity_rf.pkl')

# Load data
X_test = pd.read_csv('processed/X_test_scaled.csv')
y_test = pd.read_csv('processed/y_test.csv').squeeze()

# Predict
y_pred = model.predict(X_test)

# Accuracy
acc = accuracy_score(y_test, y_pred)
print(f" Accuracy: {acc:.2%}\n")
sys.stdout.flush()

# Classification report
print(" Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()

# Save and show image
output_path = "processed/confusion_matrix.png"
plt.savefig(output_path)
plt.show(block=True)
plt.close()

print(f" Evaluation complete. Confusion matrix saved to '{output_path}'")

# Automatically open image on Windows
if platform.system() == "Windows":
    subprocess.run(["start", output_path], shell=True)
