# backend/utils/config.py
import os
from dotenv import load_dotenv

load_dotenv()

# -------------------------
# API Keys
# -------------------------
DB_CLIENT_ID = os.getenv("DB_CLIENT_ID")
DB_CLIENT_SECRET = os.getenv("DB_CLIENT_SECRET")
OPENWEATHER_KEY = os.getenv("OPENWEATHER_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

# -------------------------
# Paths
# -------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(MODELS_DIR, "model.json")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")

# Ensure models directory exists
os.makedirs(MODELS_DIR, exist_ok=True)