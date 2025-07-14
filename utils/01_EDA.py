import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os

print("📁 Current Working Directory:", os.getcwd())

# Load dataset
df = pd.read_csv("C:/Users/deepi/OneDrive/Documents/netflix-popularity-predictor/data/netflix_titles.csv")

# Ensure plot folders
os.makedirs("plots/movies", exist_ok=True)
os.makedirs("plots/tvshows", exist_ok=True)

# =========================
# 🎬 Movies Analysis
# =========================
df_movies = df[df['type'] == 'Movie'].copy()
df_movies['genre'] = df_movies['listed_in'].str.split(',').str[0]
df_movies['duration_minutes'] = df_movies['duration'].str.extract(r'(\d+)').astype(float)
df_movies['is_popular'] = ((df_movies['release_year'] >= 2015) & (df_movies['duration_minutes'] >= 90)).astype(int)

# Generate plots for Movies
plt.figure(figsize=(10, 6))
sns.countplot(y='genre', data=df_movies, order=df_movies['genre'].value_counts().index[:10])
plt.title("Top 10 Genre Distribution (Movies)")
plt.tight_layout()
plt.savefig("plots/movies/genre_distribution.png")
plt.close()

plt.figure(figsize=(12, 6))
sns.histplot(df_movies['release_year'], bins=30, kde=True)
plt.title("Release Year Distribution (Movies)")
plt.tight_layout()
plt.savefig("plots/movies/release_year_distribution.png")
plt.close()

plt.figure(figsize=(10, 6))
sns.boxplot(x='is_popular', y='duration_minutes', data=df_movies.dropna(subset=['duration_minutes']))
plt.title("Duration vs Popularity (Movies Only)")
plt.xlabel("Is Popular (1=True, 0=False)")
plt.ylabel("Duration (minutes)")
plt.tight_layout()
plt.savefig("plots/movies/duration_vs_popularity.png")
plt.close()

corr_features = ['release_year', 'duration_minutes', 'is_popular']
plt.figure(figsize=(8, 6))
sns.heatmap(df_movies[corr_features].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap (Movies)")
plt.tight_layout()
plt.savefig("plots/movies/correlation_heatmap.png")
plt.close()

plt.figure(figsize=(14, 6))
sns.countplot(x='genre', hue='is_popular', data=df_movies, order=df_movies['genre'].value_counts().index[:10])
plt.title("Popularity Across Top 10 Genres (Movies)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("plots/movies/popularity_across_genres.png")
plt.close()

fig = px.scatter(df_movies, x='release_year', y='duration_minutes', color='is_popular',
                 hover_data=['title', 'genre'],
                 title="Release Year vs Duration Colored by Popularity (Movies)")
fig.write_html("plots/movies/release_vs_duration_popularity.html")

print("✅ Movies analysis complete and saved in 'plots/movies'.")

# =========================
# 📺 TV Shows Analysis
# =========================
df_tv = df[df['type'] == 'TV Show'].copy()
df_tv['genre'] = df_tv['listed_in'].str.split(',').str[0]
df_tv['seasons'] = df_tv['duration'].str.extract(r'(\d+)').astype(float)
df_tv['is_popular'] = ((df_tv['release_year'] >= 2015) & (df_tv['seasons'] >= 2)).astype(int)

# Generate plots for TV Shows
plt.figure(figsize=(10, 6))
sns.countplot(y='genre', data=df_tv, order=df_tv['genre'].value_counts().index[:10])
plt.title("Top 10 Genre Distribution (TV Shows)")
plt.tight_layout()
plt.savefig("plots/tvshows/genre_distribution.png")
plt.close()

plt.figure(figsize=(12, 6))
sns.histplot(df_tv['release_year'], bins=30, kde=True)
plt.title("Release Year Distribution (TV Shows)")
plt.tight_layout()
plt.savefig("plots/tvshows/release_year_distribution.png")
plt.close()

plt.figure(figsize=(10, 6))
sns.boxplot(x='is_popular', y='seasons', data=df_tv.dropna(subset=['seasons']))
plt.title("Seasons vs Popularity (TV Shows Only)")
plt.xlabel("Is Popular (1=True, 0=False)")
plt.ylabel("Number of Seasons")
plt.tight_layout()
plt.savefig("plots/tvshows/seasons_vs_popularity.png")
plt.close()

corr_features_tv = ['release_year', 'seasons', 'is_popular']
plt.figure(figsize=(8, 6))
sns.heatmap(df_tv[corr_features_tv].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap (TV Shows)")
plt.tight_layout()
plt.savefig("plots/tvshows/correlation_heatmap.png")
plt.close()

plt.figure(figsize=(14, 6))
sns.countplot(x='genre', hue='is_popular', data=df_tv, order=df_tv['genre'].value_counts().index[:10])
plt.title("Popularity Across Top 10 Genres (TV Shows)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("plots/tvshows/popularity_across_genres.png")
plt.close()

fig = px.scatter(df_tv, x='release_year', y='seasons', color='is_popular',
                 hover_data=['title', 'genre'],
                 title="Release Year vs Seasons Colored by Popularity (TV Shows)")
fig.write_html("plots/tvshows/release_vs_seasons_popularity.html")

print("✅ TV Shows analysis complete and saved in 'plots/tvshows'.")

print("""
✅ EDA complete for both Movies and TV Shows.
✅ All plots organized neatly under 'plots/movies' and 'plots/tvshows'.
✅ Ready for preprocessing and ML model building next.
""")
