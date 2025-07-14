# ============================================
# imdb_scraper.py
# Scrapes trending TV titles from IMDb Charts
# and generates a file for popularity prediction
# ============================================

import requests
from bs4 import BeautifulSoup
import pandas as pd
import random
import os

# IMDb trending TV URL
IMDB_TRENDING_URL = "https://www.imdb.com/chart/tvmeter/"

headers = {
    "User-Agent": "Mozilla/5.0"
}

print("🌐 Scraping trending titles from IMDb...")
response = requests.get(IMDB_TRENDING_URL, headers=headers)

if response.status_code != 200:
    print(f"❌ Failed to fetch data from IMDb. Status code: {response.status_code}")
    exit(1)

soup = BeautifulSoup(response.text, 'html.parser')

# Extract title names from the chart table
titles = []
for tag in soup.select(".ipc-title__text"):
    name = tag.get_text(strip=True)
    
    # Skip headers or navigation items
    if "IMDb" in name or "Most popular" in name:
        continue
    
    # Remove ranking numbers like "1. ", "2. " etc.
    clean_name = name.lstrip("1234567890. ").strip()
    
    # Avoid duplicates and blanks
    if clean_name and clean_name not in titles:
        titles.append(clean_name)

# Limit to top 10
titles = titles[:10]

# Generate dummy feature values
data = []
for title in titles:
    data.append({
        "title": title,
        "release_year": random.choice([2020, 2021, 2022, 2023, 2024, 2025]),
        "duration_minutes": random.randint(40, 150),
        "genre_encoded": random.randint(0, 15)  # Simulated genre encoding
    })

df = pd.DataFrame(data)

# Save to CSV
os.makedirs("data", exist_ok=True)
df.to_csv("data/new_netflix_titles.csv", index=False)

print("✅ Saved scraped titles to data/new_netflix_titles.csv")
print(df.head())
