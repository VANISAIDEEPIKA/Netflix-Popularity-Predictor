
# ====================================
# 03_Modeling.py
# Train and Save ML Model
# ====================================

import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load scaled training and testing data
X_train = pd.read_csv('processed/X_train_scaled.csv')
X_test = pd.read_csv('processed/X_test_scaled.csv')
y_train = pd.read_csv('processed/y_train.csv').squeeze()
y_test = pd.read_csv('processed/y_test.csv').squeeze()

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy: {accuracy:.4f}\n")
print("📊 Classification Report:")
print(classification_report(y_test, y_pred))
print("📉 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save model
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/netflix_popularity_rf.pkl')

print("✅ Model saved as 'models/netflix_popularity_rf.pkl'")
