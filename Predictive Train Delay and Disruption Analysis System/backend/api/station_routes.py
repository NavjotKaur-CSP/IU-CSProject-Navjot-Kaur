# backend/api/station_routes.py
import requests
from fastapi import APIRouter, HTTPException, Query, Depends
from sqlalchemy.orm import Session
from database import get_db

router = APIRouter()

@router.get("/stations")
def search_stations(query: str = Query(..., min_length=1)):
    """
    Proxy endpoint to Transport.rest autocomplete
    """
    try:
        url = f"https://v6.db.transport.rest/locations?query={query}"
        r = requests.get(url, timeout=8)

        if r.status_code != 200:
            raise HTTPException(status_code=502, detail="Provider error")

        data = r.json()
        results = []

        for item in data[:20]:   # only top 20
            results.append({
                "id": item.get("id"),
                "name": item.get("name"),
                "type": item.get("type"),
                "latitude": item.get("location", {}).get("latitude"),
                "longitude": item.get("location", {}).get("longitude"),
            })

        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))