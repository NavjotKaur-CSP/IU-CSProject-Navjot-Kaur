# backend/ml/train_model.py
import os
import math
import joblib
import time
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dateutil import parser
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# CONFIG
HIST_CSV = os.environ.get("HIST_CSV", "train_history_10000.csv")
MODEL_DIR = os.path.join(os.getcwd(), "models")
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "xgb_model.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")
ENCODERS_PATH = os.path.join(MODEL_DIR, "encoders.joblib")

# Helper: fetch station coords
def station_coords(station_id):
    try:
        r = requests.get(f"https://v6.db.transport.rest/locations/{station_id}", timeout=6)
        if r.status_code == 200:
            j = r.json()
            lat = j.get("location", {}).get("latitude")
            lon = j.get("location", {}).get("longitude")
            return lat, lon
    except Exception:
        pass
    return None, None

# Helper: fetch weather hourly series from Open-Meteo for a lat/lon and date range
def fetch_weather_hourly(lat, lon, start_iso, end_iso):
    if lat is None or lon is None:
        return {}
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&hourly=temperature_2m,precipitation,windspeed_10m,weathercode"
        "&timezone=UTC"
        f"&start_date={start_iso[:10]}&end_date={end_iso[:10]}"
    )
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            return r.json().get("hourly", {})
    except Exception:
        pass
    return {}

# Map a timestamp to nearest hourly index in weather data
def weather_features_for_time(weather_hourly, ts):
    # weather_hourly has arrays and 'time' list as ISO strings in UTC
    if not weather_hourly or "time" not in weather_hourly:
        return {"temp": np.nan, "precip": np.nan, "wind": np.nan, "wcode": np.nan}
    times = weather_hourly["time"]
    # convert to pandas datetime
    times_dt = pd.to_datetime(times)
    # find nearest index
    ts_utc = pd.to_datetime(ts).tz_convert("UTC") if hasattr(ts, "tzinfo") else pd.to_datetime(ts).tz_localize("UTC")
    idx = (np.abs(times_dt - ts_utc)).argmin()
    return {
        "temp": weather_hourly.get("temperature_2m", [np.nan])[idx],
        "precip": weather_hourly.get("precipitation", [np.nan])[idx],
        "wind": weather_hourly.get("windspeed_10m", [np.nan])[idx],
        "wcode": weather_hourly.get("weathercode", [np.nan])[idx],
    }

# Simple event-count feature function (OPTIONAL): provide google calendar events_count(date, station_name) -> int
def events_count_for_time(calendar_client, start_dt, end_dt, keywords=None):
    # If you implement google calendar client, return number of events between start_dt and end_dt
    # For training time, we will set 0 (placeholder) unless you integrate events.
    return 0

# Load CSV
df = pd.read_csv(HIST_CSV, low_memory=False)
print("Loaded historical rows:", df.shape)

# === Normalise columns (detect common names) ===
cols = [c.lower() for c in df.columns]
# detect delay column
delay_col = None
for cand in ("delay_minutes", "actual_delay", "actual_delay_minutes", "delay", "delay_min"):
    if cand in cols:
        delay_col = df.columns[cols.index(cand)]
        break
if delay_col is None:
    raise SystemExit("Could not find a delay column in CSV. Rename to one of delay_minutes / actual_delay / delay / delay_min")

# from/to ids or names
from_col = None; to_col = None
for cand in ("from_station_id", "from_id", "from_station", "from_station_name", "from"):
    if cand in cols:
        from_col = df.columns[cols.index(cand)]; break
for cand in ("to_station_id", "to_id", "to_station", "to_station_name", "to"):
    if cand in cols:
        to_col = df.columns[cols.index(cand)]; break

# scheduled/planned departure time column
time_col = None
for cand in ("scheduled_time", "scheduled", "planned_departure", "departure_time", "departure", "planneddeparture"):
    if cand in cols:
        time_col = df.columns[cols.index(cand)]; break

# If no explicit station ids, try to use text names
print("Detected columns:", "delay:", delay_col, "from:", from_col, "to:", to_col, "time:", time_col)

# Keep only rows with valid delay
df = df[~df[delay_col].isnull()].copy()
df[delay_col] = pd.to_numeric(df[delay_col], errors="coerce")
df = df.dropna(subset=[delay_col])
df[delay_col] = df[delay_col].astype(float)

# Prepare feature rows
rows = []
cache_weather = {}
cache_coords = {}
start_date = None
end_date = None
for idx, r in df.iterrows():
    try:
        origin = str(r[from_col]) if from_col else ""
        dest = str(r[to_col]) if to_col else ""
        # time
        if time_col and not pd.isna(r[time_col]):
            try:
                t = parser.parse(str(r[time_col]))
            except Exception:
                t = pd.to_datetime(r[time_col], errors="coerce")
        else:
            # fallback to a 'date' or 'timestamp' column, or use today
            t = pd.to_datetime(r.get("date", datetime.utcnow()))
        # record date range for weather fetch
        if start_date is None or t < start_date: start_date = t
        if end_date is None or t > end_date: end_date = t
        # get coords for origin and dest (cache)
        if origin not in cache_coords:
            lat, lon = station_coords(origin)  # tries transport.rest lookup
            cache_coords[origin] = (lat, lon)
            time.sleep(0.05)
        if dest not in cache_coords:
            lat2, lon2 = station_coords(dest)
            cache_coords[dest] = (lat2, lon2)
            time.sleep(0.05)
        origin_lat, origin_lon = cache_coords[origin]
        dest_lat, dest_lon = cache_coords[dest]
        # fetch weather hourly (cache by day)
        key_o = (origin_lat, origin_lon, t.date())
        key_d = (dest_lat, dest_lon, t.date())
        if key_o not in cache_weather:
            if origin_lat is not None and origin_lon is not None:
                cache_weather[key_o] = fetch_weather_hourly(origin_lat, origin_lon, (t - timedelta(days=1)).isoformat(), (t + timedelta(days=1)).isoformat())
            else:
                cache_weather[key_o] = {}
        if key_d not in cache_weather:
            if dest_lat is not None and dest_lon is not None:
                cache_weather[key_d] = fetch_weather_hourly(dest_lat, dest_lon, (t - timedelta(days=1)).isoformat(), (t + timedelta(days=1)).isoformat())
            else:
                cache_weather[key_d] = {}
        # get features for time
        wf_o = weather_features_for_time(cache_weather[key_o], t)
        wf_d = weather_features_for_time(cache_weather[key_d], t)
        row = {
            "origin": origin, "dest": dest,
            "hour": t.hour, "weekday": t.weekday(),
            "origin_lat": origin_lat, "origin_lon": origin_lon,
            "dest_lat": dest_lat, "dest_lon": dest_lon,
            "temp_o": wf_o["temp"], "precip_o": wf_o["precip"], "wind_o": wf_o["wind"],
            "temp_d": wf_d["temp"], "precip_d": wf_d["precip"], "wind_d": wf_d["wind"],
            "delay": float(r[delay_col])
        }
        rows.append(row)
    except Exception as e:
        continue

train_df = pd.DataFrame(rows)
print("Prepared training rows:", train_df.shape)

# Drop rows with missing delay
train_df = train_df.dropna(subset=["delay"])
# Fillna for weather numeric features
train_df[["temp_o","precip_o","wind_o","temp_d","precip_d","wind_d"]] = train_df[["temp_o","precip_o","wind_o","temp_d","precip_d","wind_d"]].fillna(0)

# Encode categorical: origin/dest
le_origin = LabelEncoder()
le_dest = LabelEncoder()
train_df["origin_enc"] = le_origin.fit_transform(train_df["origin"].astype(str))
train_df["dest_enc"] = le_dest.fit_transform(train_df["dest"].astype(str))

# Features and target
FEATURES = ["hour","weekday","origin_enc","dest_enc","origin_lat","origin_lon","dest_lat","dest_lon",
            "temp_o","precip_o","wind_o","temp_d","precip_d","wind_d"]
X = train_df[FEATURES].fillna(0).values
y = train_df["delay"].values

# scaler
scaler = StandardScaler()
Xs = scaler.fit_transform(X)

# train/test split
X_train, X_val, y_train, y_val = train_test_split(Xs, y, test_size=0.15, random_state=42)

# XGBoost regressor (small default)
from xgboost import XGBRegressor

model = XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    n_jobs=4
)

# XGBoost >= 3.0: register early stopping via set_params or callbacks
model.set_params(early_stopping_rounds=20)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)
# eval
preds = model.predict(X_val)
mae = mean_absolute_error(y_val, preds)
rmse = math.sqrt(mean_squared_error(y_val, preds))
print(f"Validation MAE: {mae:.3f}, RMSE: {rmse:.3f}")

# Save model and scaler + encoders
joblib.dump(model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)
joblib.dump({"origin_le": le_origin, "dest_le": le_dest}, ENCODERS_PATH)
print("Saved model to", MODEL_PATH)
print("Saved scaler to", SCALER_PATH)
print("Saved encoders to", ENCODERS_PATH)