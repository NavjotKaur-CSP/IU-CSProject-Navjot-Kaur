# routers/prediction.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from ml.predict import predict_and_explain
from typing import Optional

router = APIRouter(prefix="/predict", tags=["predict"])

class PredictRequest(BaseModel):
    train_id: str
    station: str
    scheduled_time: str  # ISO datetime string
    weather_temp: Optional[float] = None
    weather_rain_mm: Optional[float] = None

@router.post("/", summary="Predict delay and return SHAP top features")
def predict(req: PredictRequest):
    try:
        payload = req.dict()
        res = predict_and_explain(payload)
        return {"status": "ok", "result": res}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
