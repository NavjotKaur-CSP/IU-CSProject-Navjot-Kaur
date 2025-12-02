from sqlalchemy import Column, Integer, String, Float, DateTime
from sqlalchemy.sql import func
from database import Base

class TrainSearch(Base):
    __tablename__ = "train_searches"
    
    id = Column(Integer, primary_key=True, index=True)
    from_station = Column(String, index=True)
    to_station = Column(String, index=True)
    search_query = Column(String)  # What user searched for
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class SelectedTrain(Base):
    __tablename__ = "selected_trains"
    
    id = Column(Integer, primary_key=True, index=True)
    train_id = Column(String, index=True)
    from_station = Column(String)
    to_station = Column(String)
    departure_time = Column(String)
    arrival_time = Column(String)
    predicted_delay = Column(Integer)
    transport_type = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=func.now())