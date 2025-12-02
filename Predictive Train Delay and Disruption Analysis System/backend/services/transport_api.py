import requests
from typing import Optional, List, Dict, Any

BASE = "https://v6.db.transport.rest"

# Helper to do GET requests and return JSON or None
def _get(url: str, params=None, headers=None, accept_json=True) -> Optional[Any]:
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        if resp.status_code >= 400:
            # return JSON if available, else text
            try:
                return {"error": resp.status_code, "body": resp.json()}
            except Exception:
                return {"error": resp.status_code, "body": resp.text}
        if accept_json:
            return resp.json()
        return resp.text
    except Exception as e:
        return None

def search_locations(query: str) -> Optional[List[Dict]]:
    """
    GET /locations?query=...
    returns list of simplified location dicts
    """
    url = f"{BASE}/locations"
    params = {"query": query, "types": "stop,station,stopplace", "results": 10}
    data = _get(url, params=params)
    if data is None:
        return None
    out = []
    for item in data:
        loc = {
            "id": str(item.get("id") or item.get("evaId") or item.get("stopId") or item.get("provider") or ""),
            "name": item.get("name"),
            "type": item.get("type"),
            "latitude": item.get("location", {}).get("latitude"),
            "longitude": item.get("location", {}).get("longitude"),
            "raw": item
        }
        out.append(loc)
    return out

def get_departures_for_station(station_id: str, duration: int = 60) -> Optional[Dict]:
    """
    GET /stops/{id}/departures?duration=60
    """
    url = f"{BASE}/stops/{station_id}/departures"
    params = {"duration": duration}
    data = _get(url, params=params)
    if data is None:
        return None
    return data

def get_journeys_by_ids(from_id: str, to_id: str, results: int = 10) -> Optional[List[Dict]]:
    """
    Uses /journeys?from=...&to=... to fetch journey details
    """
    url = f"{BASE}/journeys"
    params = {"from": from_id, "to": to_id, "results": results}
    data = _get(url, params=params)
    if data is None:
        return None
    # data expected to have "journeys"
    if isinstance(data, dict) and "journeys" in data:
        return data["journeys"]
    # if the API returned directly a list
    if isinstance(data, list):
        return data
    return data