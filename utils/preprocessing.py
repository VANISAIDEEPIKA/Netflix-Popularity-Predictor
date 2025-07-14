# utils/preprocessing.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ================================
# 🎭 Encode Genre Column
# ================================
def encode_genre_column(df, col_name='genre'):
    """
    Encodes the genre column using Label Encoding.
    Returns updated DataFrame and encoder object.
    """
    le = LabelEncoder()
    df['genre_encoded'] = le.fit_transform(df[col_name].fillna('Unknown'))
    return df, le

# ================================
# 📏 Scale Features
# ================================
def scale_features(X_train, X_test=None):
    """
    Standardizes features using StandardScaler.
    Returns scaled training (and optional test) sets + scaler object.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled, scaler
    
    return X_train_scaled, scaler

# ================================
# 🕵️ Validate Required Columns
# ================================
def check_required_columns(df, required_cols):
    """
    Validates if required columns exist in the DataFrame.
    Raises error if any are missing.
    """
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"❌ Missing columns in dataset: {missing_cols}")
    print("✅ All required columns are present.")
if __name__ == "__main__":
    import pandas as pd

    # 🧪 Sample test data
    sample_data = pd.DataFrame({
        "genre": ["Action", "Drama", "Horror", "Action"],
        "release_year": [2020, 2021, 2022, 2023],
        "duration_minutes": [100, 90, 80, 110]
    })

    # ✅ Encode genre
    encoded_df, le = encode_genre_column(sample_data)
    print("\n🎭 Genre Encoding Test:")
    print(encoded_df[['genre', 'genre_encoded']])

    # ✅ Scale features
    X = encoded_df[["release_year", "duration_minutes", "genre_encoded"]]
    X_scaled, scaler = scale_features(X)
    print("\n📊 Scaled Features:")
    print(X_scaled)

    # ✅ Check required columns
    print("\n🧩 Required Columns Check:")
    check_required_columns(encoded_df, ["release_year", "duration_minutes", "genre_encoded"])
