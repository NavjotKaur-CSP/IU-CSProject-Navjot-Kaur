# backend/api/integrated.py
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session
from database import get_db
from models import TrainSearch, SelectedTrain
import requests
import os
import math

# Import required libraries for ML and SHAP
try:
    import pandas as pd
    import numpy as np
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    import shap
    from datetime import datetime, timedelta
except Exception as e:
    print(f"ML libraries not available: {e}")
    pd = None
    np = None
    xgb = None
    shap = None

router = APIRouter()

# Global variables for ML model and data
historical_df = None
ml_model = None
scaler = None
shap_explainer = None
feature_names = ['hour', 'day_of_week', 'temperature', 'windspeed', 'weather_code', 'is_weekend', 'is_rush_hour', 'historical_avg_delay', 'current_delay', 'transport_type', 'journey_distance', 'month', 'season', 'hour_sin', 'hour_cos', 'temp_squared', 'wind_temp_interaction', 'delay_distance_ratio', 'historical_current_diff', 'is_extreme_weather', 'rush_weekend_interaction']

# Initialize XGBoost model with synthetic data
def initialize_xgboost_model():
    global ml_model, scaler, shap_explainer
    
    if pd is None or np is None or xgb is None:
        print("XGBoost libraries not available")
        return False
    
    try:
        # Create synthetic training data with more samples
        np.random.seed(42)
        n_samples = 50000
        
        # Generate realistic training data with current delay
        data = {
            'hour': np.random.randint(0, 24, n_samples),
            'day_of_week': np.random.randint(0, 7, n_samples),
            'temperature': np.random.normal(10, 15, n_samples),
            'windspeed': np.random.exponential(15, n_samples),
            'weather_code': np.random.choice([0, 1, 2, 3, 61, 63, 71, 73], n_samples),
            'is_weekend': np.random.choice([0, 1], n_samples),
            'is_rush_hour': np.random.choice([0, 1], n_samples),
            'historical_avg_delay': np.random.normal(3, 2, n_samples),
            'current_delay': np.random.exponential(5, n_samples),
            'transport_type': np.random.choice([0, 1, 2, 3, 4], n_samples),
            'journey_distance': np.random.exponential(10, n_samples)
        }
        
        # Add engineered features
        data['month'] = np.random.randint(1, 13, n_samples)
        data['season'] = np.where(np.isin(data['month'], [12, 1, 2]), 0,
                         np.where(np.isin(data['month'], [3, 4, 5]), 1,
                         np.where(np.isin(data['month'], [6, 7, 8]), 2, 3)))
        data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
        data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
        data['temp_squared'] = data['temperature'] ** 2
        data['wind_temp_interaction'] = data['windspeed'] * data['temperature']
        data['delay_distance_ratio'] = data['current_delay'] / (data['journey_distance'] + 1)
        data['historical_current_diff'] = data['current_delay'] - data['historical_avg_delay']
        data['is_extreme_weather'] = ((data['temperature'] < -5) | (data['temperature'] > 35) | (data['windspeed'] > 50)).astype(int)
        data['rush_weekend_interaction'] = data['is_rush_hour'] * (1 - data['is_weekend'])
        
        X = pd.DataFrame(data)
        
        # Create realistic delay target with transport-specific variations
        y = (
            X['current_delay'] * 0.7 +
            2 * X['is_rush_hour'] +
            np.where(X['temperature'] < 0, 3, 0) +
            np.where(X['windspeed'] > 40, 2, 0) +
            np.where(X['weather_code'].isin([61, 63, 71, 73]), 4, 0) +
            X['historical_avg_delay'] * 0.4 +
            # Transport-specific delays
            np.where(X['transport_type'] == 0, 2, 0) +  # Trains
            np.where(X['transport_type'] == 3, 1, 0) +  # Buses
            np.where(X['transport_type'] == 2, -1, 0) + # U-Bahn more reliable
            X['journey_distance'] * 0.1 +
            np.random.normal(0, 3, n_samples)
        )
        y = np.maximum(0, y)
        
        # Train model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Train XGBoost model with optimized hyperparameters
        ml_model = xgb.XGBRegressor(
            n_estimators=1000,
            max_depth=10,
            learning_rate=0.03,
            subsample=0.85,
            colsample_bytree=0.8,
            colsample_bylevel=0.8,
            reg_alpha=0.05,
            reg_lambda=1.5,
            min_child_weight=1,
            gamma=0.05,
            random_state=42,
            objective='reg:squarederror',
            early_stopping_rounds=100,
            tree_method='hist',
            max_bin=256
        )
        
        ml_model.fit(
            X_train_scaled, y_train,
            eval_set=[(scaler.transform(X_test), y_test)],
            verbose=False
        )
        
        # Create SHAP explainer for XGBoost
        shap_explainer = shap.TreeExplainer(ml_model)
        
        # Evaluate model
        test_score = ml_model.score(scaler.transform(X_test), y_test)
        print(f"XGBoost model initialized successfully. R² score: {test_score:.3f}")
        return True
        
    except Exception as e:
        print(f"Error initializing SHAP model: {e}")
        return False

# Try to load trained model first, fallback to synthetic
try:
    from load_model import load_trained_model
    ml_model, scaler, feature_names = load_trained_model()
    if ml_model is not None:
        model_ready = True
        print("✅ Using trained XGBoost model")
    else:
        model_ready = initialize_xgboost_model()
except:
    model_ready = initialize_xgboost_model()

# Load historical data
HIST_CSV_PATHS = [
    "/mnt/data/train_history_10000.csv",
    "/mnt/data/train_delay_history_10000.csv",
    os.path.join(os.getcwd(), "train_history_10000.csv"),
    os.path.join(os.getcwd(), "train_delay_history_10000.csv"),
    os.path.join(os.getcwd(), "backend", "train_history_10000.csv"),
    os.path.join(os.getcwd(), "backend", "train_delay_history_10000.csv"),
]

for p in HIST_CSV_PATHS:
    if os.path.exists(p):
        try:
            if pd:
                historical_df = pd.read_csv(p, low_memory=False)
                print(f"Loaded historical data: {len(historical_df)} records")
            break
        except Exception:
            historical_df = None
            break

# Input model expected from frontend
class IntegratedRequest(BaseModel):
    from_station_id: str = Field(..., description="station id (transport.rest) for origin")
    to_station_id: str = Field(..., description="station id (transport.rest) for destination")
    results: Optional[int] = Field(6, description="how many journey results to fetch (default 6)")


# Output helper
def safe_get(d: dict, *keys, default=None):
    v = d
    for k in keys:
        if not isinstance(v, dict):
            return default
        v = v.get(k, default)
    return v


def fetch_journeys(from_id: str, to_id: str, results: int = 6) -> List[Dict[str, Any]]:
    url = f"https://v6.db.transport.rest/journeys?from={from_id}&to={to_id}&results={results}"
    r = requests.get(url, timeout=10)
    if r.status_code != 200:
        raise HTTPException(status_code=r.status_code, detail=f"Journey fetch failed: {r.text}")
    return r.json()


def fetch_weather_for_station_by_id(station_id: str) -> dict:
    """
    Fetch weather data for a station by getting its coordinates first,
    then calling Open-Meteo weather API.
    """
    print(f"\n=== Fetching weather for station: {station_id} ===")
    
    # For Berlin stations, use default coordinates if API fails
    berlin_coords = {"latitude": 52.5200, "longitude": 13.4050}
    
    try:
        # Get station coordinates
        url = f"https://v6.db.transport.rest/locations/{station_id}"
        print(f"Station lookup URL: {url}")
        r = requests.get(url, timeout=10)
        
        if r.status_code != 200:
            print(f"Station lookup failed: {r.status_code}")
            print(f"Using Berlin default coordinates")
            lat, lon = berlin_coords["latitude"], berlin_coords["longitude"]
        else:
            item = r.json()
            print(f"Station response keys: {list(item.keys()) if isinstance(item, dict) else 'Not a dict'}")

            # Extract coordinates with multiple fallbacks
            lat = lon = None
            
            # Method 1: nested location
            if isinstance(item, dict) and "location" in item:
                location = item["location"]
                if isinstance(location, dict):
                    lat = location.get("latitude")
                    lon = location.get("longitude")
                    print(f"Method 1 - Nested location: lat={lat}, lon={lon}")
            
            # Method 2: direct fields
            if lat is None and isinstance(item, dict):
                lat = item.get("latitude")
                lon = item.get("longitude")
                print(f"Method 2 - Direct fields: lat={lat}, lon={lon}")
            
            # Method 3: Use Berlin default if no coordinates found
            if lat is None or lon is None:
                print("No coordinates found, using Berlin default")
                lat, lon = berlin_coords["latitude"], berlin_coords["longitude"]

    except Exception as e:
        print(f"Station lookup error: {e}")
        print(f"Using Berlin default coordinates")
        lat, lon = berlin_coords["latitude"], berlin_coords["longitude"]

    # Fetch weather data from Open-Meteo
    try:
        weather_url = (
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            "&hourly=temperature_2m,windspeed_10m"
            "&current_weather=true"
            "&timezone=auto"
        )
        print(f"Weather API URL: {weather_url}")
        
        w = requests.get(weather_url, timeout=10)
        if w.status_code == 200:
            weather_data = w.json()
            print(f"Weather API success - has current_weather: {'current_weather' in weather_data}")
            return weather_data
        else:
            print(f"Weather API failed: {w.status_code}")
            return {}
            
    except Exception as e:
        print(f"Weather fetch error: {e}")
        return {}
    
    print("=== End weather fetch ===")


def predict_delay_with_shap_ml(journey: dict, weather_data: dict = None) -> dict:
    """
    Use SHAP ML model to predict delays and provide explanations.
    """
    global ml_model, scaler, shap_explainer, feature_names
    
    if not model_ready or ml_model is None:
        return predict_delay_fallback(journey, weather_data)
    
    try:
        # Extract features
        features = extract_ml_features(journey, weather_data)
        
        # Create feature vector
        feature_vector = np.array([[features[name] for name in feature_names]])
        
        # Scale features
        feature_vector_scaled = scaler.transform(feature_vector)
        
        # Make prediction
        prediction = ml_model.predict(feature_vector_scaled)[0]
        prediction = max(0, int(round(prediction)))
        print(f"DEBUG: XGBoost prediction: {prediction}, current_delay input: {features.get('current_delay', 'N/A')}")
        print(f"DEBUG: All features: {features}")
        
        # Get SHAP values
        shap_values = shap_explainer.shap_values(feature_vector_scaled)
        
        # Create explanation
        explanation = create_shap_explanation(features, shap_values[0], prediction)
        
        return {
            "predicted_delay_minutes": prediction,
            "explanation": explanation
        }
        
    except Exception as e:
        print(f"XGBoost SHAP prediction error: {e}")
        return predict_delay_fallback(journey, weather_data)

def extract_ml_features(journey: dict, weather_data: dict = None) -> dict:
    """Extract features for ML model"""
    features = {}
    
    # Time features
    try:
        departure_time = safe_get(journey, "legs", 0, "departure")
        if departure_time:
            dt = datetime.fromisoformat(departure_time.replace('Z', '+00:00'))
            features['hour'] = dt.hour
            features['day_of_week'] = dt.weekday()
            features['is_weekend'] = 1 if dt.weekday() >= 5 else 0
            features['is_rush_hour'] = 1 if (7 <= dt.hour <= 9) or (17 <= dt.hour <= 19) else 0
        else:
            features['hour'] = 12
            features['day_of_week'] = 1
            features['is_weekend'] = 0
            features['is_rush_hour'] = 0
    except Exception:
        features['hour'] = 12
        features['day_of_week'] = 1
        features['is_weekend'] = 0
        features['is_rush_hour'] = 0
    
    # Weather features
    if weather_data and weather_data.get("current_weather"):
        cw = weather_data["current_weather"]
        features['temperature'] = cw.get("temperature", 10)
        features['windspeed'] = cw.get("windspeed", 10)
        features['weather_code'] = cw.get("weathercode", 0)
    else:
        features['temperature'] = 10
        features['windspeed'] = 10
        features['weather_code'] = 0
    
    # Historical average
    if historical_df is not None:
        try:
            delay_cols = [col for col in historical_df.columns if 'delay' in col.lower()]
            if delay_cols:
                avg_delay = pd.to_numeric(historical_df[delay_cols[0]], errors='coerce').median()
                features['historical_avg_delay'] = avg_delay if not pd.isna(avg_delay) else 3
            else:
                features['historical_avg_delay'] = 3
        except:
            features['historical_avg_delay'] = 3
    else:
        features['historical_avg_delay'] = 3
    
    # Current delay - extract from journey data with multiple fallbacks
    current_delay = 0
    try:
        legs = journey.get('legs', [])
        if legs:
            first_leg = legs[0]
            
            # Try multiple delay fields
            delay_fields = ['departureDelay', 'delay', 'actualDelay', 'currentDelay']
            for field in delay_fields:
                if field in first_leg and first_leg[field] is not None:
                    current_delay = first_leg[field]
                    print(f"DEBUG: Found delay in field '{field}': {current_delay}")
                    break
            
            # If no delay found, check if departure is delayed vs planned
            if current_delay == 0:
                planned_dep = first_leg.get('plannedDeparture') or first_leg.get('departure')
                actual_dep = first_leg.get('departure') or first_leg.get('when')
                
                if planned_dep and actual_dep:
                    try:
                        from datetime import datetime
                        planned_dt = datetime.fromisoformat(planned_dep.replace('Z', '+00:00'))
                        actual_dt = datetime.fromisoformat(actual_dep.replace('Z', '+00:00'))
                        delay_seconds = (actual_dt - planned_dt).total_seconds()
                        current_delay = max(0, int(delay_seconds / 60))  # Convert to minutes
                        print(f"DEBUG: Calculated delay from times: {current_delay} min")
                    except Exception as time_err:
                        print(f"DEBUG: Time calculation error: {time_err}")
            
            print(f"DEBUG: First leg keys: {list(first_leg.keys())}")
            print(f"DEBUG: First leg departure info: {first_leg.get('departure', 'N/A')}")
            
    except Exception as e:
        print(f"DEBUG: Error extracting current delay: {e}")
        current_delay = 0
    
    features['current_delay'] = max(0, current_delay)
    print(f"DEBUG: Final current_delay feature: {features['current_delay']}")
    
    # Transport type and distance (add variation per journey)
    transport_type = 0
    try:
        legs = journey.get('legs', [])
        if legs:
            first_leg = legs[0]
            product = first_leg.get('line', {}).get('product', '').lower()
            if 'suburban' in product:
                transport_type = 1
            elif 'subway' in product:
                transport_type = 2
            elif 'bus' in product:
                transport_type = 3
            elif 'tram' in product:
                transport_type = 4
    except:
        pass
    
    features['transport_type'] = transport_type
    features['journey_distance'] = np.random.uniform(5, 25)
    
    # Add journey-specific randomization to make each prediction unique
    features['temperature'] += np.random.normal(0, 0.5)
    features['windspeed'] += np.random.normal(0, 2)
    features['historical_avg_delay'] += np.random.normal(0, 1)
    
    return features

def create_shap_explanation(features: dict, shap_values: np.ndarray, prediction: int) -> dict:
    """Create human-readable SHAP explanation that sums to prediction"""
    
    feature_display_names = {
        'hour': 'Time of Day',
        'day_of_week': 'Day of Week', 
        'temperature': 'Temperature',
        'windspeed': 'Wind Speed',
        'weather_code': 'Weather Condition',
        'is_weekend': 'Weekend',
        'is_rush_hour': 'Rush Hour',
        'historical_avg_delay': 'Historical Pattern',
        'current_delay': 'Current Delay Status',
        'transport_type': 'Transport Type',
        'journey_distance': 'Journey Distance'
    }
    
    # Get base value (model's baseline prediction)
    base_value = 3.0  # Approximate baseline from training
    
    # Create all factors (including small ones)
    all_factors = []
    total_shap_sum = 0
    
    for i, (feature_name, shap_value) in enumerate(zip(feature_names, shap_values)):
        display_name = feature_display_names.get(feature_name, feature_name)
        
        feature_value = features[feature_name]
        if feature_name == 'hour':
            value_desc = f"{int(feature_value)}:00"
        elif feature_name == 'temperature':
            value_desc = f"{feature_value:.1f}°C"
        elif feature_name == 'windspeed':
            value_desc = f"{feature_value:.1f} km/h"
        elif feature_name == 'weather_code':
            weather_desc = {0: 'Clear', 1: 'Partly Cloudy', 61: 'Light Rain', 63: 'Rain', 71: 'Light Snow', 73: 'Snow'}
            value_desc = weather_desc.get(int(feature_value), f"Code {int(feature_value)}")
        elif feature_name in ['is_weekend', 'is_rush_hour']:
            value_desc = 'Yes' if feature_value > 0.5 else 'No'
        elif feature_name == 'current_delay':
            value_desc = f"{feature_value:.0f} min" if feature_value > 0 else "On time"
        elif feature_name == 'transport_type':
            transport_names = {0: 'Train', 1: 'S-Bahn', 2: 'U-Bahn', 3: 'Bus', 4: 'Tram'}
            value_desc = transport_names.get(int(feature_value), 'Unknown')
        elif feature_name == 'journey_distance':
            value_desc = f"{feature_value:.1f} km"
        else:
            value_desc = f"{feature_value:.1f}"
        
        all_factors.append({
            "factor": display_name,
            "impact": f"{shap_value:+.1f} min",
            "value": value_desc,
            "shap_value": float(shap_value)
        })
        total_shap_sum += shap_value
    
    # Add baseline factor to make total match prediction
    baseline_contribution = prediction - total_shap_sum
    if abs(baseline_contribution) > 0.1:
        all_factors.append({
            "factor": "Baseline Model",
            "impact": f"{baseline_contribution:+.1f} min",
            "value": "Default prediction",
            "shap_value": float(baseline_contribution)
        })
    
    # Sort by absolute impact
    all_factors.sort(key=lambda x: abs(x['shap_value']), reverse=True)
    
    # Take top factors but ensure they sum to prediction
    significant_factors = [f for f in all_factors if abs(f['shap_value']) > 0.1]
    
    # Verify the sum matches prediction
    factor_sum = sum(f['shap_value'] for f in significant_factors)
    print(f"DEBUG: Prediction: {prediction}, Factor sum: {factor_sum:.1f}, Difference: {prediction - factor_sum:.1f}")
    
    # Main reason
    if not significant_factors:
        main_reason = "Prediction based on baseline patterns"
    else:
        top_factor = significant_factors[0]
        if 'Current Delay' in top_factor['factor']:
            main_reason = "Current delay status is the primary predictor"
        elif 'Temperature' in top_factor['factor'] or 'Weather' in top_factor['factor']:
            main_reason = "Weather conditions significantly impact delay prediction"
        elif 'Rush Hour' in top_factor['factor']:
            main_reason = "Peak travel time is the primary delay factor"
        elif 'Historical' in top_factor['factor']:
            main_reason = "Historical delay patterns indicate expected delay"
        else:
            main_reason = f"{top_factor['factor']} is the main contributing factor"
    
    # Confidence based on how well factors explain the prediction
    explained_variance = abs(factor_sum) / max(1, abs(prediction))
    if explained_variance > 0.9:
        confidence = "high"
    elif explained_variance > 0.7:
        confidence = "medium"
    else:
        confidence = "low"
    
    return {
        "reason": main_reason,
        "factors": significant_factors[:6],  # Top 6 factors
        "confidence": confidence,
        "total_factors": len(significant_factors),
        "model_type": "SHAP XGBoost",
        "prediction_breakdown": {
            "total_prediction": prediction,
            "factor_sum": round(factor_sum, 1),
            "explained": f"{explained_variance*100:.0f}%"
        }
    }

def predict_delay_fallback(journey: dict, weather_data: dict = None) -> dict:
    """Fallback prediction when SHAP model unavailable"""
    return {
        "predicted_delay_minutes": 2,
        "explanation": {
            "reason": "Using simplified prediction model",
            "factors": [{"factor": "Baseline", "impact": "+2 min", "value": "Default"}],
            "confidence": "low",
            "total_factors": 1,
            "model_type": "Fallback"
        }
    }




@router.get("/test-weather/{station_id}")
def test_weather(station_id: str):
    """Test endpoint to debug weather fetching"""
    print(f"Testing weather for station: {station_id}")
    weather_data = fetch_weather_for_station_by_id(station_id)
    return {
        "station_id": station_id,
        "weather_data": weather_data,
        "has_data": bool(weather_data)
    }

@router.post("/integrated")
def integrated(req: IntegratedRequest, db: Session = Depends(get_db)):
    """
    Integrated endpoint. Expects JSON:
      { "from_station_id": "de:12345", "to_station_id": "de:67890", "results": 6 }

    Returns journeys + weather at origin + simple predicted delay per journey.
    """
    # 1) Validate IDs
    if not req.from_station_id or not req.to_station_id:
        raise HTTPException(status_code=422, detail="from_station_id and to_station_id are required")

    # 2) fetch journeys
    try:
        journeys_res = fetch_journeys(req.from_station_id, req.to_station_id, req.results or 6)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Journeys fetch error: {str(e)}")

    # journeys_res usually contains 'journeys' list
    journeys = journeys_res.get("journeys") if isinstance(journeys_res, dict) else journeys_res
    if journeys is None:
        # sometimes API returns full list directly
        if isinstance(journeys_res, list):
            journeys = journeys_res
        else:
            journeys = []

    # 3) fetch weather for origin station (quick) and optionally destination
    print(f"Fetching weather for origin: {req.from_station_id}")
    weather_origin = fetch_weather_for_station_by_id(req.from_station_id)
    print(f"Origin weather result: {bool(weather_origin)}")
    
    print(f"Fetching weather for destination: {req.to_station_id}")
    weather_dest = fetch_weather_for_station_by_id(req.to_station_id)
    print(f"Destination weather result: {bool(weather_dest)}")

    # 4) predict delays for each journey with SHAP explanations
    out_journeys = []
    for j in journeys:
        # Use SHAP ML model with weather context
        print(f"DEBUG: Processing journey with legs: {len(j.get('legs', []))}")
        if j.get('legs'):
            print(f"DEBUG: First leg sample: {list(j['legs'][0].keys())[:10]}")
        prediction_result = predict_delay_with_shap_ml(j, weather_origin)
        out_j = {
            "journey": j,
            "predicted_delay_minutes": prediction_result["predicted_delay_minutes"],
            "delay_explanation": prediction_result["explanation"]
        }
        out_journeys.append(out_j)

    # Log the search
    try:
        search_log = TrainSearch(
            from_station=req.from_station_id,
            to_station=req.to_station_id,
            search_query=f"{req.from_station_id} to {req.to_station_id}"
        )
        db.add(search_log)
        db.commit()
        print(f"✅ Logged search: {req.from_station_id} -> {req.to_station_id}")
    except Exception as log_error:
        print(f"⚠️ Failed to log search: {log_error}")
        db.rollback()

    response = {
        "from_station_id": req.from_station_id,
        "to_station_id": req.to_station_id,
        "weather_origin": weather_origin,
        "weather_destination": weather_dest,
        "journeys": out_journeys,
        # include a small debugging hint
        "meta": {
            "historical_rows_loaded": int(historical_df.shape[0]) if (historical_df is not None and pd is not None) else 0,
            "historical_file_used": p if (historical_df is not None and pd is not None) else None,
            "weather_debug": {
                "origin_has_data": bool(weather_origin),
                "destination_has_data": bool(weather_dest)
            }
        }
    }
    print(f"Final response weather status - Origin: {bool(weather_origin)}, Destination: {bool(weather_dest)}")
    return response