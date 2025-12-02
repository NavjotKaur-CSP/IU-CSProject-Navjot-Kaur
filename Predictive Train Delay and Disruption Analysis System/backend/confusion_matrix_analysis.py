import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def create_enhanced_features(n_samples=10000):
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
    
    # Create realistic delay target
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
        np.where(X['transport_type'] == 0, 3, 0) +
        np.where(X['transport_type'] == 3, 2, 0) +
        np.where(X['transport_type'] == 2, -1, 0) +
        np.where(X['season'] == 0, 2, 0) +
        X['hour_sin'] * X['temperature'] * 0.1 +
        X['hour_cos'] * X['windspeed'] * 0.05 +
        np.random.normal(0, 2, n_samples)
    )
    y = np.maximum(0, y)
    
    return X, y

def delay_to_category(delay):
    """Convert continuous delay to categories"""
    if delay <= 2:
        return 0  # "On Time"
    elif delay <= 5:
        return 1  # "Minor Delay"
    elif delay <= 15:
        return 2  # "Moderate Delay"
    else:
        return 3  # "Major Delay"

def create_confusion_matrix():
    """Create and visualize confusion matrix"""
    print("Creating dataset...")
    X, y_continuous = create_enhanced_features(10000)
    
    # Convert to categories
    y_categorical = np.array([delay_to_category(delay) for delay in y_continuous])
    
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.3, random_state=42)
    
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Training XGBoost classifier...")
    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.8,
        random_state=42,
        objective='multi:softprob',
        num_class=4
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Category labels
    labels = ['On Time\n(≤2 min)', 'Minor Delay\n(2-5 min)', 'Moderate Delay\n(5-15 min)', 'Major Delay\n(>15 min)']
    
    # Create visualization
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Number of Predictions'})
    
    plt.title('Train Delay Prediction - Confusion Matrix', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Delay Category', fontsize=12)
    plt.ylabel('Actual Delay Category', fontsize=12)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Calculate accuracy
    accuracy = np.sum(np.diag(cm)) / np.sum(cm)
    print(f"\nModel Accuracy: {accuracy:.3f}")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=labels))
    
    # Print confusion matrix with percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    print("\nConfusion Matrix (Percentages):")
    print("Actual \\ Predicted", end="")
    for label in labels:
        print(f"\t{label.split()[0]}", end="")
    print()
    
    for i, label in enumerate(labels):
        print(f"{label.split()[0]}\t\t", end="")
        for j in range(len(labels)):
            print(f"{cm_percent[i,j]:.1f}%\t", end="")
        print()
    
    return model, scaler, cm, accuracy

if __name__ == "__main__":
    model, scaler, cm, accuracy = create_confusion_matrix()
    print(f"\nFinal Classification Accuracy: {accuracy:.3f}")