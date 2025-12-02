# generate_train_history.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

random.seed(42)
np.random.seed(42)

N = 10000

stations = [
    "Berlin Hbf","Berlin Gesundbrunnen","Berlin Ostbahnhof","Berlin Südkreuz",
    "Berlin Alexanderplatz","Berlin Neukölln","Berlin Lichtenberg","Berlin Charlottenburg",
    "Berlin Spandau","Berlin Westend","Berlin Zoo","Berlin Friedrichstraße",
    "Berlin Wartenberg","Berlin Tempelhof","Berlin Treptower Park","Berlin Schöneweide",
    "Berlin Köpenick","Berlin Pankow","Berlin Blankenburg","Berlin Adlershof",
    "Berlin Marzahn","Berlin Grunewald","Berlin Dahlem","Berlin Britz","Berlin Wilmersdorf"
]

transport_types = ["ICE","RE","S-Bahn","U-Bahn","Tram","Bus"]

start = datetime(2023,1,1)
end = datetime(2025,1,1)
total_seconds = int((end-start).total_seconds())
scheduled_times = [start + timedelta(seconds=int(random.random()*total_seconds)) for _ in range(N)]
scheduled_times = pd.to_datetime(sorted(scheduled_times))

from_stations = [random.choice(stations) for _ in range(N)]
to_stations = []
for s in from_stations:
    dest = random.choice(stations)
    while dest == s:
        dest = random.choice(stations)
    to_stations.append(dest)

probs = [0.12,0.10,0.25,0.18,0.20,0.15]
types = list(np.random.choice(transport_types, size=N, p=probs))

def make_train_id(t):
    if t in ("ICE","RE"): return f"{t}{random.randint(100,999)}"
    if t=="S-Bahn": return f"S{random.randint(1,9)}{random.randint(100,999)}"
    if t=="U-Bahn": return f"U{random.randint(1,9)}{random.randint(10,99)}"
    if t=="Tram": return f"T{random.randint(1,9)}{random.randint(10,99)}"
    return f"B{random.randint(1,99)}{random.randint(10,99)}"

train_ids = [make_train_id(t) for t in types]

rand = np.random.rand(N)
delay_minutes = np.zeros(N, dtype=int)
for i in range(N):
    r = rand[i]
    if r < 0.60:
        delay_minutes[i] = int(np.random.choice(range(-5,2)))
    elif r < 0.90:
        delay_minutes[i] = int(np.random.poisson(3) + 1)
    elif r < 0.99:
        delay_minutes[i] = int(np.random.randint(11,61))
    else:
        delay_minutes[i] = int(np.random.randint(61,241))

actual_times = scheduled_times + pd.to_timedelta(delay_minutes, unit="m")

month_avg = {1:0,2:1,3:5,4:9,5:14,6:17,7:19,8:18,9:14,10:9,11:4,12:1}
temps=[]; rains=[]
for t in scheduled_times:
    m = t.month; avg = month_avg.get(m,10)
    temp = round(np.random.normal(avg,4),1)
    rain = round(max(0, np.random.exponential(0.5) - (0.1 if temp>15 else 0)),2)
    temps.append(temp); rains.append(rain)

df = pd.DataFrame({
    "train_id": train_ids,
    "transport_type": types,
    "from_station": from_stations,
    "to_station": to_stations,
    "scheduled_time": scheduled_times,
    "actual_time": actual_times,
    "delay_minutes": delay_minutes,
    "weather_temp_c": temps,
    "weather_rain_mm": rains
})
df["weekday"] = df["scheduled_time"].dt.day_name()
df["hour"] = df["scheduled_time"].dt.hour

# pick path
out_path = "/mnt/data/train_history_10000.csv"
if not os.path.isdir("/mnt/data"):
    out_path = os.path.join(os.getcwd(), "train_history_10000.csv")

df.to_csv(out_path, index=False)
print("Saved:", out_path)
print("Rows:", len(df))