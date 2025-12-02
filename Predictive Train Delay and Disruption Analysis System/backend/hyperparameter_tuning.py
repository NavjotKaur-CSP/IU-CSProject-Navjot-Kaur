import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def tune_hyperparameters():
    """Hyperparameter tuning to maximize R² score"""
    
    # Load historical data
    try:
        df = pd.read_csv('train_history_10000.csv')
        print(f"Loaded {len(df)} historical records")
        
        # Process data
        df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
        df['hour'] = df['datetime'].dt.hour
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_rush_hour'] = ((df['hour'].between(7, 9)) | (df['hour'].between(17, 19))).astype(int)
        
        # Map transport types
        transport_map = {'ICE': 0, 'IC': 1, 'RE': 2, 'RB': 3, 'S-Bahn': 4, 'U-Bahn': 5}
        df['transport_type'] = df['product'].map(transport_map).fillna(0)
        
        # Add synthetic features
        np.random.seed(42)
        n = len(df)
        df['temperature'] = np.random.normal(10, 15, n)
        df['windspeed'] = np.random.exponential(15, n)
        df['weather_code'] = np.random.choice([0, 1, 2, 3, 61, 63, 71, 73], n)
        df['journey_distance'] = np.random.exponential(10, n)
        
        # Historical average delay
        station_avg_delay = df.groupby('station')['delay_minutes'].mean().to_dict()
        df['historical_avg_delay'] = df['station'].map(station_avg_delay)
        
        # Features and target
        features = ['hour', 'day_of_week', 'temperature', 'windspeed', 'weather_code', 
                   'is_weekend', 'is_rush_hour', 'historical_avg_delay', 'transport_type', 'journey_distance']
        
        X = df[features].fillna(0)
        y = np.clip(df['delay_minutes'].fillna(0), -10, 120)
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Hyperparameter grid
    param_grid = {
        'n_estimators': [500, 1000, 1500],
        'max_depth': [8, 10, 12],
        'learning_rate': [0.01, 0.03, 0.05],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'reg_alpha': [0, 0.1, 0.5],
        'reg_lambda': [0.5, 1.0, 2.0]
    }
    
    # Grid search
    print("Starting hyperparameter tuning...")
    xgb_model = xgb.XGBRegressor(random_state=42, objective='reg:squarederror')
    
    grid_search = GridSearchCV(
        xgb_model, 
        param_grid, 
        cv=5, 
        scoring='r2',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train_scaled, y_train)
    
    # Best model
    best_model = grid_search.best_estimator_
    
    # Evaluate
    y_pred = best_model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    print(f"\nBest Parameters: {grid_search.best_params_}")
    print(f"Best R² Score: {r2:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {np.sqrt(mse):.4f}")
    
    # Save best model
    joblib.dump(best_model, "models/best_xgb_model.joblib")
    joblib.dump(scaler, "models/best_scaler.joblib")
    
    return best_model, scaler, r2

if __name__ == "__main__":
    tune_hyperparameters()