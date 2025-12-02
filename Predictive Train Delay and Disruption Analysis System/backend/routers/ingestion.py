# routers/ingestion.py
from fastapi import APIRouter, HTTPException
from backend.services.transport_api import get_timetable, get_fasta, get_stada
from typing import Any

router = APIRouter(prefix="/ingest", tags=["ingest"])

@router.get("/timetable/{station}/{date}")
def ingest_timetable(station: str, date: str):
    try:
        data = get_timetable(station, date)
        # TODO: parse and save to DB if desired
        return {"status": "ok", "data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/fasta/{train_id}")
def ingest_fasta(train_id: str):
    try:
        data = get_fasta(train_id)
        return {"status": "ok", "data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stada/{station}")
def ingest_stada(station: str):
    try:
        data = get_stada(station)
        return {"status": "ok", "data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
