# services/weather_api.py
import os
import requests
OPENWEATHER_KEY = os.getenv("OPENWEATHER_KEY", "")

def get_weather(lat: float, lon: float) -> dict:
    if not OPENWEATHER_KEY:
        return {}
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"lat": lat, "lon": lon, "appid": OPENWEATHER_KEY, "units": "metric"}
    r = requests.get(url, params=params)
    r.raise_for_status()
    return r.json()
