import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os

# ========== 🎨 Page Configuration ==========
st.set_page_config(page_title="🎬 Netflix Show Popularity Predictor", layout="wide")
st.title("🎥 Netflix Popularity Predictor")
st.markdown("##### Built with ❤️ by **VANI SAI DEEPIKA** ")

# ========== 🎯 User Choice: Upload or Demo ==========
mode = st.radio("Choose Mode", ["📁 Upload Your Dataset", "🧪 Try Demo Mode"])

if mode == "📁 Upload Your Dataset":
    uploaded_movies = st.file_uploader("Upload your `netflix_movies_detailed_up_to_2025.csv` file", type=["csv"], key="movies")
    uploaded_shows = st.file_uploader("Upload your `netflix_tv_shows_detailed_up_to_2025.csv` file", type=["csv"], key="shows")
    if uploaded_movies and uploaded_shows:
        movies_df = pd.read_csv(uploaded_movies)
        shows_df = pd.read_csv(uploaded_shows)
    else:
        st.info("👈 Please upload both CSVs to proceed.")
        st.stop()
else:
    movies_df = pd.read_csv("netflix_movies_detailed_up_to_2025.csv")
    shows_df = pd.read_csv("netflix_tv_shows_detailed_up_to_2025.csv")
    st.success("✅ Loaded sample datasets for demo mode!")

# ========== 🔄 Merge Data ==========
common_cols = ['title', 'listed_in', 'release_year', 'duration']
movies_df = movies_df[common_cols]
shows_df = shows_df[common_cols]
df = pd.concat([movies_df, shows_df], ignore_index=True)

# ========== 🧹 Preprocessing ==========
df['genre'] = df['listed_in'].str.split(',').str[0].fillna('Unknown')
df['duration_minutes'] = df['duration'].str.extract(r'(\d+)').astype(float)
df['is_popular'] = ((df['release_year'] >= 2015) & (df['duration_minutes'] >= 90)).astype(int)

# ========== 🧾 Dataset Preview ==========
with st.expander("📄 Dataset Preview"):
    st.dataframe(df[['title', 'genre', 'release_year', 'duration_minutes', 'is_popular']].head(10))

# ========== 📊 Genre vs Popularity ==========
st.subheader("📊 Genre vs Popularity")
fig1 = px.histogram(df, x="genre", color="is_popular", barmode="group",
                    title="Genre vs Popularity", height=400)
st.plotly_chart(fig1)

# ========== 📈 Release Year Trend ==========
st.subheader("📈 Release Year Distribution")
fig2 = px.histogram(df, x="release_year", nbins=20, title="Release Year Trend", height=400)
st.plotly_chart(fig2)

# ========== 🔍 Feature Encoding ==========
features = ['genre', 'release_year', 'duration_minutes']
target = 'is_popular'
df_model = df[features + [target]].dropna()
le = LabelEncoder()
df_model['genre'] = le.fit_transform(df_model['genre'])
X = df_model[features]
y = df_model[target]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ========== 🚀 Train Model ==========
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ========== 🧠 Evaluation ==========
st.subheader("⚙️ Model Performance")
col1, col2 = st.columns(2)
with col1:
    acc = accuracy_score(y_test, y_pred)
    st.metric("🎯 Accuracy", f"{acc*100:.2f}%")
with col2:
    st.write("📜 Classification Report")
    st.text(classification_report(y_test, y_pred))

st.write("📉 Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig3, ax = plt.subplots()
sns.heatmap(cm, annot=True, cmap="Purples", fmt="d", ax=ax)
st.pyplot(fig3)

# ========== 🔥 Feature Importance ==========
st.subheader("🔥 Feature Importance")
importance = model.feature_importances_
imp_df = pd.DataFrame({
    "Feature": features,
    "Importance": importance
}).sort_values(by="Importance", ascending=False)
fig4 = px.bar(imp_df, x='Feature', y='Importance', color='Importance', title="Which Features Matter Most?")
st.plotly_chart(fig4)

# ========== 💾 Save Model ==========
if st.button("💾 Save Model"):
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/netflix_rf.pkl")
    joblib.dump(scaler, "models/netflix_scaler.pkl")
    st.success("✅ Model saved successfully!")

# ========== 🧡 Footer ==========
st.markdown("---")
st.markdown("✨ *Made with 💜 for internships & backend dominance by VANISAIDEEPIKA*")
