from fastapi import APIRouter
from pydantic import BaseModel
import requests

router = APIRouter()

# ---------------------------
#  /api/locations
# ---------------------------
@router.get("/locations")
def get_locations(query: str):
    url = f"https://v6.db.transport.rest/locations?query={query}"
    res = requests.get(url)

    if res.status_code != 200:
        return {"error": res.text}

    data = res.json()
    results = []

    for item in data:
        results.append({
            "id": item.get("id"),
            "name": item.get("name"),
            "latitude": item.get("location", {}).get("latitude"),
            "longitude": item.get("location", {}).get("longitude"),
            "type": item.get("type")
        })

    return results


# ---------------------------
#  /api/journeys
# ---------------------------
class JourneyRequest(BaseModel):
    from_name: str
    to_name: str
    results: int = 10

@router.post("/journeys")
def get_journeys(req: JourneyRequest):

    # 1 >>> GET FROM STATION ID
    from_url = f"https://v6.db.transport.rest/locations?query={req.from_name}"
    from_res = requests.get(from_url).json()
    if not from_res:
        return {"error": "Invalid from station"}
    from_id = from_res[0]["id"]

    # 2 >>> GET TO STATION ID
    to_url = f"https://v6.db.transport.rest/locations?query={req.to_name}"
    to_res = requests.get(to_url).json()
    if not to_res:
        return {"error": "Invalid to station"}
    to_id = to_res[0]["id"]

    # 3 >>> REQUEST JOURNEYS
    journeys_url = f"https://v6.db.transport.rest/journeys?from={from_id}&to={to_id}&results={req.results}"
    journeys_res = requests.get(journeys_url)

    full = journeys_res.json()

# Extract only journeys list
    journeys = full.get("journeys", [])

    return journeys