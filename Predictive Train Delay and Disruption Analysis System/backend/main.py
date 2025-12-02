from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.integrated import router as integrated_router
from api.station_routes import router as station_router
from api.journey_routes import router as journey_router
from database import create_tables, engine

app = FastAPI(title="Predictive Train Delay API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create tables on startup
@app.on_event("startup")
def startup_event():
    try:
        create_tables()
        print("✅ Database tables created")
    except Exception as e:
        print(f"⚠️ Database connection failed: {e}")

# Include APIs
app.include_router(station_router, prefix="/api")
app.include_router(integrated_router, prefix="/api")
app.include_router(journey_router, prefix="/api")

@app.get("/")
def root():
    return {"message": "Predictive Train Delay API Running with PostgreSQL"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "message": "API is running with PostgreSQL"}

@app.get("/tables")
def check_tables():
    from sqlalchemy import text
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT tablename FROM pg_tables WHERE schemaname = 'public';"))
            tables = [row[0] for row in result.fetchall()]
            return {"tables": tables}
    except Exception as e:
        return {"error": str(e)}