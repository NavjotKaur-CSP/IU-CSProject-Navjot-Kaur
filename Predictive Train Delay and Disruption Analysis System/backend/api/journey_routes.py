from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel
from database import get_db
from models import TrainSearch, SelectedTrain

router = APIRouter()

class SearchRequest(BaseModel):
    from_station: str
    to_station: str
    search_query: str

class TrainSelection(BaseModel):
    train_id: str
    from_station: str
    to_station: str
    departure_time: str
    arrival_time: str
    predicted_delay: int
    transport_type: str

@router.post("/search")
def log_search(search: SearchRequest, db: Session = Depends(get_db)):
    """Log train search details"""
    search_log = TrainSearch(
        from_station=search.from_station,
        to_station=search.to_station,
        search_query=search.search_query
    )
    db.add(search_log)
    db.commit()
    return {"message": "Search logged"}

@router.post("/select-train")
def log_train_selection(train: TrainSelection, db: Session = Depends(get_db)):
    """Log selected train with delay details"""
    selected_train = SelectedTrain(
        train_id=train.train_id,
        from_station=train.from_station,
        to_station=train.to_station,
        departure_time=train.departure_time,
        arrival_time=train.arrival_time,
        predicted_delay=train.predicted_delay,
        transport_type=train.transport_type
    )
    db.add(selected_train)
    db.commit()
    return {"message": "Train selection logged"}

@router.get("/searches")
def get_searches(db: Session = Depends(get_db)):
    """Get recent searches"""
    searches = db.query(TrainSearch).order_by(TrainSearch.created_at.desc()).limit(50).all()
    return searches

@router.get("/selected-trains")
def get_selected_trains(db: Session = Depends(get_db)):
    """Get recent train selections with delays"""
    trains = db.query(SelectedTrain).order_by(SelectedTrain.created_at.desc()).limit(50).all()
    return trains