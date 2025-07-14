
# =======================================
# 02_Preprocessing.py
# Netflix Popularity Prediction Pipeline
# =======================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Load cleaned dataset
DATA_PATH = 'data/cleaned_netflix_dataset.csv'
print(f"✅ Loading dataset: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

# Basic checks
print(df.info())
print(df.describe())
print(df.isnull().sum())

# Drop missing rows (should be none, but safe)
df = df.dropna()

# Check required columns
required_columns = ['genre_encoded', 'release_year', 'duration_minutes', 'is_popular']
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    raise ValueError(f"❌ Missing columns in dataset: {missing_columns}")

# Features & Target
features = ['release_year', 'duration_minutes', 'genre_encoded']
target = 'is_popular'

X = df[features]
y = df[target]

# Show target distribution
print("🎯 Target class distribution:")
print(y.value_counts(normalize=True))

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Scale Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save processed data
os.makedirs('processed', exist_ok=True)
pd.DataFrame(X_train_scaled, columns=features).to_csv('processed/X_train_scaled.csv', index=False)
pd.DataFrame(X_test_scaled, columns=features).to_csv('processed/X_test_scaled.csv', index=False)
y_train.to_csv('processed/y_train.csv', index=False)
y_test.to_csv('processed/y_test.csv', index=False)

# Save scaler for reuse
joblib.dump(scaler, 'processed/scaler.pkl')

print("✅ Preprocessing complete. Data saved in 'processed' folder.")
print("📁 Saved Files:")
print("- processed/X_train_scaled.csv")
print("- processed/X_test_scaled.csv")
print("- processed/y_train.csv")
print("- processed/y_test.csv")
print("- processed/scaler.pkl")
