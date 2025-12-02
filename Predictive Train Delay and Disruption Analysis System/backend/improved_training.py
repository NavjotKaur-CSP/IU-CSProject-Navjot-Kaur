import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import joblib

def create_enhanced_features(n_samples=100000):
    """Create enhanced synthetic dataset with more realistic patterns"""
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
    
    # Additional complex features
    data['temp_wind_combined'] = np.sqrt(data['temperature']**2 + data['windspeed']**2)
    data['delay_momentum'] = data['current_delay'] * data['journey_distance']
    data['weather_severity'] = np.where(data['weather_code'] > 50, 2, np.where(data['weather_code'] > 10, 1, 0))
    data['peak_hour_factor'] = np.where(np.isin(data['hour'], [7, 8, 9, 17, 18, 19]), 2, 1)
    
    X = pd.DataFrame(data)
    
    # More realistic target with complex interactions
    y = (
        X['current_delay'] * 0.6 +
        X['historical_avg_delay'] * 0.3 +
        X['is_rush_hour'] * 3 +
        X['peak_hour_factor'] * 1.5 +
        np.where(X['temperature'] < 0, 4, 0) +
        np.where(X['windspeed'] > 40, 3, 0) +
        X['weather_severity'] * 2 +
        X['delay_momentum'] * 0.05 +
        X['temp_wind_combined'] * 0.1 +
        # Transport-specific patterns
        np.where(X['transport_type'] == 0, 2.5, 0) +  # Trains
        np.where(X['transport_type'] == 3, 1.5, 0) +  # Buses
        np.where(X['transport_type'] == 2, -0.5, 0) + # U-Bahn
        # Seasonal effects
        np.where(X['season'] == 0, 1.5, 0) +  # Winter
        np.where(X['season'] == 2, -0.5, 0) + # Summer
        # Non-linear interactions
        X['hour_sin'] * X['temperature'] * 0.2 +
        X['delay_distance_ratio'] * 5 +
        np.random.normal(0, 2, n_samples)  # Reduced noise
    )
    y = np.maximum(0, y)
    
    return X, y

def train_optimized_model():
    """Train model with hyperparameter optimization"""
    print("Creating enhanced dataset...")
    X, y = create_enhanced_features(100000)
    
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Hyperparameter tuning
    print("Tuning hyperparameters...")
    param_grid = {
        'n_estimators': [800, 1200],
        'max_depth': [8, 12],
        'learning_rate': [0.02, 0.05],
        'subsample': [0.8, 0.9],
        'colsample_bytree': [0.8, 0.9]
    }
    
    xgb_model = xgb.XGBRegressor(
        random_state=42,
        objective='reg:squarederror',
        tree_method='hist'
    )
    
    grid_search = GridSearchCV(
        xgb_model, param_grid, 
        cv=3, scoring='r2', 
        n_jobs=-1, verbose=1
    )
    
    grid_search.fit(X_train_scaled, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV R² score: {grid_search.best_score_:.4f}")
    
    # Final model with best parameters
    best_model = grid_search.best_estimator_
    
    # Evaluate
    train_score = best_model.score(X_train_scaled, y_train)
    test_score = best_model.score(X_test_scaled, y_test)
    
    print(f"Train R² score: {train_score:.4f}")
    print(f"Test R² score: {test_score:.4f}")
    
    # Save model
    joblib.dump(best_model, 'models/optimized_xgb_model.joblib')
    joblib.dump(scaler, 'models/optimized_scaler.joblib')
    
    return best_model, scaler, test_score

if __name__ == "__main__":
    model, scaler, score = train_optimized_model()
    print(f"Final optimized model R² score: {score:.4f}")