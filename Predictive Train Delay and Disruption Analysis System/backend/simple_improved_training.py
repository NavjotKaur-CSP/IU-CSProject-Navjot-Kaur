import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import joblib
import os

def create_enhanced_features(n_samples=50000):
    """Create enhanced synthetic dataset"""
    np.random.seed(42)
    
    # Base features
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
        'journey_distance': np.random.exponential(10, n_samples),
        'month': np.random.randint(1, 13, n_samples)
    }
    
    # Enhanced feature engineering
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
    
    # More realistic target with stronger relationships
    y = (
        X['current_delay'] * 0.8 +
        X['historical_avg_delay'] * 0.5 +
        X['is_rush_hour'] * 4 +
        X['rush_weekend_interaction'] * 3 +
        np.where(X['temperature'] < 0, 5, 0) +
        np.where(X['windspeed'] > 40, 4, 0) +
        np.where(X['weather_code'].isin([61, 63, 71, 73]), 6, 0) +
        X['delay_distance_ratio'] * 8 +
        X['wind_temp_interaction'] * 0.05 +
        X['historical_current_diff'] * 0.3 +
        X['is_extreme_weather'] * 5 +
        # Transport-specific patterns
        np.where(X['transport_type'] == 0, 3, 0) +  # Trains
        np.where(X['transport_type'] == 3, 2, 0) +  # Buses
        np.where(X['transport_type'] == 2, -1, 0) + # U-Bahn
        # Seasonal effects
        np.where(X['season'] == 0, 2, 0) +  # Winter
        # Cyclical patterns
        X['hour_sin'] * X['temperature'] * 0.1 +
        X['hour_cos'] * X['windspeed'] * 0.05 +
        np.random.normal(0, 1.5, n_samples)  # Less noise
    )
    y = np.maximum(0, y)
    
    return X, y

def train_improved_model():
    """Train improved model with better features"""
    print("Creating enhanced dataset...")
    X, y = create_enhanced_features(50000)
    
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Training optimized XGBoost model...")
    model = xgb.XGBRegressor(
        n_estimators=1500,
        max_depth=12,
        learning_rate=0.02,
        subsample=0.85,
        colsample_bytree=0.8,
        colsample_bylevel=0.8,
        reg_alpha=0.1,
        reg_lambda=2.0,
        min_child_weight=1,
        gamma=0.1,
        random_state=42,
        objective='reg:squarederror',
        tree_method='hist',
        max_bin=512
    )
    
    # Train with validation set for early stopping
    model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_test_scaled, y_test)],
        early_stopping_rounds=100,
        verbose=False
    )
    
    # Evaluate
    train_score = model.score(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)
    
    print(f"Train R² score: {train_score:.4f}")
    print(f"Test R² score: {test_score:.4f}")
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save model
    joblib.dump(model, 'models/improved_xgb_model.joblib')
    joblib.dump(scaler, 'models/improved_scaler.joblib')
    
    print("Model saved successfully!")
    return model, scaler, test_score

if __name__ == "__main__":
    model, scaler, score = train_improved_model()
    print(f"Final improved model R² score: {score:.4f}")