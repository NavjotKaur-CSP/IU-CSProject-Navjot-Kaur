import pandas as pd
from sklearn.preprocessing import StandardScaler
from backend.services.model_loader import save_scaler

def simple_feature_engineer(df):
    df = df.copy()

    if "scheduled_time" in df.columns:
        df["scheduled_time"] = pd.to_datetime(df["scheduled_time"])
        df["hour"] = df["scheduled_time"].dt.hour
        df["dayofweek"] = df["scheduled_time"].dt.dayofweek
        df = df.drop(columns=["scheduled_time", "train_id", "station"], errors="ignore")

    return df

def scale_features(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    save_scaler(scaler)
    return X_scaled