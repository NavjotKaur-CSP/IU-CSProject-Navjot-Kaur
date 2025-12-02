import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
from sqlalchemy import create_engine
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime

load_dotenv()

def load_data_from_db():
    """Load training data from PostgreSQL database"""
    DATABASE_URL = os.getenv("DATABASE_URL")
    engine = create_engine(DATABASE_URL)
    
    try:
        # Load journey searches data
        query = """
        SELECT 
            from_station,
            to_station,
            train_id,
            predicted_delay,
            EXTRACT(hour FROM created_at) as hour,
            EXTRACT(dow FROM created_at) as day_of_week,
            created_at
        FROM journey_searches
        ORDER BY created_at DESC
        """
        
        df = pd.read_sql(query, engine)
        print(f"Loaded {len(df)} records from database")
        return df
        
    except Exception as e:
        print(f"Error loading from database: {e}")
        return None

def load_historical_data():
    """Load and process historical train data"""
    print("Loading historical train data...")
    
    try:
        # Load the CSV file
        df = pd.read_csv('train_history_10000.csv')
        print(f"Loaded {len(df)} historical records")
        
        # Process datetime
        df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
        df['hour'] = df['datetime'].dt.hour
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_rush_hour'] = ((df['hour'].between(7, 9)) | (df['hour'].between(17, 19))).astype(int)
        
        # Map transport types
        transport_map = {'ICE': 0, 'IC': 1, 'RE': 2, 'RB': 3, 'S-Bahn': 4, 'U-Bahn': 5}
        df['transport_type'] = df['product'].map(transport_map).fillna(0)
        
        # Add synthetic weather features (since not in historical data)
        np.random.seed(42)
        n = len(df)
        df['temperature'] = np.random.normal(10, 15, n)
        df['windspeed'] = np.random.exponential(15, n)
        df['weather_code'] = np.random.choice([0, 1, 2, 3, 61, 63, 71, 73], n)
        df['journey_distance'] = np.random.exponential(10, n)
        
        # Calculate historical average delay per station
        station_avg_delay = df.groupby('station')['delay_minutes'].mean().to_dict()
        df['historical_avg_delay'] = df['station'].map(station_avg_delay)
        
        # Select features
        features = ['hour', 'day_of_week', 'temperature', 'windspeed', 'weather_code', 
                   'is_weekend', 'is_rush_hour', 'historical_avg_delay', 'transport_type', 'journey_distance']
        
        X = df[features].fillna(0)
        y = df['delay_minutes'].fillna(0)
        
        # Remove extreme outliers
        y = np.clip(y, -10, 120)  # Cap delays between -10 and 120 minutes
        
        print(f"Processed features: {features}")
        print(f"Delay range: {y.min():.1f} to {y.max():.1f} minutes")
        print(f"Average delay: {y.mean():.1f} minutes")
        
        return X, y
        
    except Exception as e:
        print(f"Error loading historical data: {e}")
        return None, None

def train_xgboost_model():
    """Train XGBoost model"""
    print("Starting XGBoost training...")
    
    # Try to load historical data first
    X, y = load_historical_data()
    
    if X is None or len(X) < 100:
        print("Historical data not available, creating synthetic data...")
        # Fallback synthetic data
        np.random.seed(42)
        n_samples = 5000
        
        data = {
            'hour': np.random.randint(0, 24, n_samples),
            'day_of_week': np.random.randint(0, 7, n_samples),
            'temperature': np.random.normal(10, 15, n_samples),
            'windspeed': np.random.exponential(15, n_samples),
            'weather_code': np.random.choice([0, 1, 2, 3, 61, 63, 71, 73], n_samples),
            'is_weekend': np.random.choice([0, 1], n_samples),
            'is_rush_hour': np.random.choice([0, 1], n_samples),
            'historical_avg_delay': np.random.normal(3, 2, n_samples),
            'transport_type': np.random.choice([0, 1, 2, 3, 4], n_samples),
            'journey_distance': np.random.exponential(10, n_samples)
        }
        
        X = pd.DataFrame(data)
        y = (
            2 * X['is_rush_hour'] +
            np.where(X['temperature'] < 0, 3, 0) +
            np.where(X['windspeed'] > 40, 2, 0) +
            np.where(X['weather_code'].isin([61, 63, 71, 73]), 4, 0) +
            X['historical_avg_delay'] * 0.4 +
            np.where(X['transport_type'] == 0, 2, 0) +
            X['journey_distance'] * 0.1 +
            np.random.normal(0, 3, n_samples)
        )
        y = np.maximum(0, y)
    else:
        print(f"Using historical data with {len(X)} records")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train XGBoost model with optimized hyperparameters
    model = xgb.XGBRegressor(
        n_estimators=1000,
        max_depth=10,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_weight=3,
        gamma=0.1,
        random_state=42,
        objective='reg:squarederror',
        early_stopping_rounds=100
    )
    
    print("Training model...")
    model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_test_scaled, y_test)],
        verbose=False
    )
    
    # Evaluate model
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model Performance:")
    print(f"MSE: {mse:.3f}")
    print(f"R² Score: {r2:.3f}")
    print(f"RMSE: {np.sqrt(mse):.3f}")
    
    # Save model and scaler
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    
    joblib.dump(model, f"{model_dir}/xgb_model.joblib")
    joblib.dump(scaler, f"{model_dir}/scaler.joblib")
    
    # Save feature names
    feature_names = list(X.columns)
    joblib.dump(feature_names, f"{model_dir}/feature_names.joblib")
    
    print(f"Model saved to {model_dir}/")
    print(f"Feature names: {feature_names}")
    
    # Create visualizations
    create_visualizations(X, y, y_pred, model, feature_names, y_test)
    
    # Save training details
    save_training_details(X, y, y_test, y_pred, mse, r2, model, feature_names)
    
    return model, scaler, feature_names

def create_visualizations(X, y, y_pred, model, feature_names, y_test):
    """Create training visualizations"""
    plt.style.use('default')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('XGBoost Model Training Analysis', fontsize=16)
    
    # 1. Delay distribution
    axes[0,0].hist(y, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0,0].set_title('Distribution of Train Delays')
    axes[0,0].set_xlabel('Delay (minutes)')
    axes[0,0].set_ylabel('Frequency')
    axes[0,0].axvline(y.mean(), color='red', linestyle='--', label=f'Mean: {y.mean():.1f}min')
    axes[0,0].legend()
    
    # 2. Actual vs Predicted
    axes[0,1].scatter(y_test, y_pred, alpha=0.6, color='green')
    axes[0,1].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    axes[0,1].set_title('Actual vs Predicted Delays')
    axes[0,1].set_xlabel('Actual Delay (minutes)')
    axes[0,1].set_ylabel('Predicted Delay (minutes)')
    
    # 3. Feature importance
    importance = model.feature_importances_
    feature_imp = pd.DataFrame({'feature': feature_names, 'importance': importance})
    feature_imp = feature_imp.sort_values('importance', ascending=True)
    
    axes[0,2].barh(feature_imp['feature'], feature_imp['importance'], color='orange')
    axes[0,2].set_title('Feature Importance')
    axes[0,2].set_xlabel('Importance')
    
    # 4. Residuals
    residuals = y_test - y_pred
    axes[1,0].scatter(y_pred, residuals, alpha=0.6, color='purple')
    axes[1,0].axhline(y=0, color='red', linestyle='--')
    axes[1,0].set_title('Residuals vs Predicted')
    axes[1,0].set_xlabel('Predicted Delay (minutes)')
    axes[1,0].set_ylabel('Residuals')
    
    # 5. Delay by transport type (if historical data available)
    try:
        df_viz = pd.read_csv('train_history_10000.csv')
        delay_by_transport = df_viz.groupby('product')['delay_minutes'].mean().sort_values(ascending=False)
        
        axes[1,1].bar(delay_by_transport.index, delay_by_transport.values, color='lightcoral')
        axes[1,1].set_title('Average Delay by Transport Type')
        axes[1,1].set_xlabel('Transport Type')
        axes[1,1].set_ylabel('Average Delay (minutes)')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        # 6. Delay by hour
        df_viz['datetime'] = pd.to_datetime(df_viz['date'] + ' ' + df_viz['time'])
        df_viz['hour'] = df_viz['datetime'].dt.hour
        delay_by_hour = df_viz.groupby('hour')['delay_minutes'].mean()
        
        axes[1,2].plot(delay_by_hour.index, delay_by_hour.values, marker='o', color='teal')
        axes[1,2].set_title('Average Delay by Hour of Day')
        axes[1,2].set_xlabel('Hour')
        axes[1,2].set_ylabel('Average Delay (minutes)')
        axes[1,2].grid(True, alpha=0.3)
        
    except Exception as e:
        axes[1,1].text(0.5, 0.5, 'Historical data\nnot available', ha='center', va='center', transform=axes[1,1].transAxes)
        axes[1,2].text(0.5, 0.5, 'Historical data\nnot available', ha='center', va='center', transform=axes[1,2].transAxes)
    
    plt.tight_layout()
    plt.savefig('models/training_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("📊 Visualizations saved to models/training_analysis.png")

def save_training_details(X, y, y_test, y_pred, mse, r2, model, feature_names):
    """Save comprehensive training details to JSON file"""
    
    # Calculate additional metrics
    mae = np.mean(np.abs(y_test - y_pred))
    rmse = np.sqrt(mse)
    
    # Feature importance
    feature_importance = dict(zip(feature_names, model.feature_importances_.tolist()))
    
    # Data statistics
    data_stats = {
        "total_samples": len(X),
        "training_samples": len(X) - len(y_test),
        "test_samples": len(y_test),
        "delay_range": [float(y.min()), float(y.max())],
        "delay_mean": float(y.mean()),
        "delay_std": float(y.std())
    }
    
    # Model parameters
    model_params = model.get_params()
    
    # Training details
    training_details = {
        "timestamp": datetime.now().isoformat(),
        "model_type": "XGBoost",
        "data_source": "Historical train data (train_history_10000.csv)",
        "performance_metrics": {
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
            "r2_score": float(r2),
            "accuracy_percentage": float((1 - mae/max(1, y.mean())) * 100)
        },
        "data_statistics": data_stats,
        "feature_importance": feature_importance,
        "model_parameters": {k: v for k, v in model_params.items() if isinstance(v, (int, float, str, bool))},
        "feature_names": feature_names,
        "prediction_samples": {
            "actual": y_test[:10].tolist(),
            "predicted": y_pred[:10].tolist()
        }
    }
    
    # Save to JSON file
    with open('models/training_details.json', 'w') as f:
        json.dump(training_details, f, indent=2)
    
    # Save to database if available
    try:
        save_to_database(training_details)
    except Exception as e:
        print(f"Could not save to database: {e}")
    
    print("💾 Training details saved to models/training_details.json")
    
def save_to_database(training_details):
    """Save training details to PostgreSQL database"""
    DATABASE_URL = os.getenv("DATABASE_URL")
    if not DATABASE_URL:
        return
        
    engine = create_engine(DATABASE_URL)
    
    # Create training_logs table if not exists
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS training_logs (
        id SERIAL PRIMARY KEY,
        timestamp TIMESTAMP DEFAULT NOW(),
        model_type VARCHAR(50),
        data_source TEXT,
        mse FLOAT,
        rmse FLOAT,
        mae FLOAT,
        r2_score FLOAT,
        accuracy_percentage FLOAT,
        total_samples INTEGER,
        feature_importance JSON,
        model_parameters JSON,
        training_details JSON
    );
    """
    
    with engine.connect() as conn:
        conn.execute(create_table_sql)
        
        # Insert training record
        insert_sql = """
        INSERT INTO training_logs (
            model_type, data_source, mse, rmse, mae, r2_score, 
            accuracy_percentage, total_samples, feature_importance, 
            model_parameters, training_details
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        metrics = training_details["performance_metrics"]
        data_stats = training_details["data_statistics"]
        
        conn.execute(insert_sql, (
            training_details["model_type"],
            training_details["data_source"],
            metrics["mse"],
            metrics["rmse"],
            metrics["mae"],
            metrics["r2_score"],
            metrics["accuracy_percentage"],
            data_stats["total_samples"],
            json.dumps(training_details["feature_importance"]),
            json.dumps(training_details["model_parameters"]),
            json.dumps(training_details)
        ))
        
        conn.commit()
    
    print("💾 Training details saved to database")

if __name__ == "__main__":
    train_xgboost_model()