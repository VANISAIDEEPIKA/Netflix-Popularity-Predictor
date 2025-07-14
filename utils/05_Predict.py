# ====================================
# 05_Predict.py
# Predict Popularity on New Data
# ====================================

import pandas as pd
import joblib
import os
import sys
import matplotlib.pyplot as plt
import subprocess

# ======================================
# 🔄 Auto-fetch new data from IMDb
# ======================================
SCRAPER_PATH = 'utils/imdb_scraper.py'
if os.path.exists(SCRAPER_PATH):
    print("🌐 Running IMDb scraper to fetch latest trending titles...")
    subprocess.run(["python", SCRAPER_PATH])
else:
    print("⚠️ IMDb scraper not found. Skipping data update.")

# ======================================
# 📦 Load Model and Scaler
# ======================================
try:
    model = joblib.load('models/netflix_popularity_rf.pkl')
    scaler = joblib.load('processed/scaler.pkl')
except FileNotFoundError as e:
    print(f"❌ Could not load model or scaler: {e}")
    sys.exit(1)

# ======================================
# 📥 Load New Data
# ======================================
NEW_DATA_PATH = 'data/new_netflix_titles.csv'
print(f"📥 Loading new data from: {NEW_DATA_PATH}")

if not os.path.exists(NEW_DATA_PATH):
    print(f"❌ File not found: {NEW_DATA_PATH}")
    print("🔧 Please make sure the file exists and has the required columns.")
    sys.exit(1)

new_df = pd.read_csv(NEW_DATA_PATH)

# ======================================
# ✅ Check Required Columns
# ======================================
required_cols = ['release_year', 'duration_minutes', 'genre_encoded']
missing_cols = [col for col in required_cols if col not in new_df.columns]
if missing_cols:
    raise ValueError(f"❌ Missing columns in input data: {missing_cols}")

# ======================================
# 🤖 Predict Popularity
# ======================================
X_new = new_df[required_cols]
X_scaled = pd.DataFrame(scaler.transform(X_new), columns=X_new.columns)
predictions = model.predict(X_scaled)
new_df['predicted_popularity'] = predictions

# Optional manual override for iconic titles
iconic_titles = ["Squid Game", "Stranger Things", "The Witcher", "Bridgerton"]
new_df.loc[new_df['title'].isin(iconic_titles), 'predicted_popularity'] = 1

# ======================================
# 💾 Save Results
# ======================================
OUTPUT_PATH = 'processed/predictions.csv'
os.makedirs('processed', exist_ok=True)
new_df.to_csv(OUTPUT_PATH, index=False)
print(f"✅ Predictions saved to: {OUTPUT_PATH}")

# ======================================
# 🔍 Preview Output
# ======================================
preview_cols = ['predicted_popularity']
if 'title' in new_df.columns:
    preview_cols.insert(0, 'title')

print("\n🔍 Preview of Predictions:")
print(new_df[preview_cols].head(10))

# ======================================
# 📊 Popularity Bar Chart
# ======================================
new_df['predicted_popularity'].value_counts().sort_index().plot(
    kind='bar',
    color=['salmon', 'mediumseagreen'],
    figsize=(6, 4)
)
plt.title("Predicted Popularity Count")
plt.xlabel("Popularity (0 = Not Popular, 1 = Popular)")
plt.ylabel("Number of Titles")
plt.xticks(ticks=[0, 1], labels=["Not Popular", "Popular"], rotation=0)
plt.tight_layout()
plt.savefig("processed/prediction_distribution.png")
plt.show()
