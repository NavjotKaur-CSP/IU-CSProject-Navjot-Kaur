import joblib
import os
import numpy as np

def load_trained_model():
    """Load the trained XGBoost model"""
    model_dir = "models"
    
    try:
        model = joblib.load(f"{model_dir}/xgb_model.joblib")
        scaler = joblib.load(f"{model_dir}/scaler.joblib")
        feature_names = joblib.load(f"{model_dir}/feature_names.joblib")
        
        print("✅ Loaded trained XGBoost model")
        return model, scaler, feature_names
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None, None, None

def predict_delay(model, scaler, feature_names, features_dict):
    """Make prediction using trained model"""
    try:
        # Create feature vector
        feature_vector = np.array([[features_dict[name] for name in feature_names]])
        
        # Scale features
        feature_vector_scaled = scaler.transform(feature_vector)
        
        # Make prediction
        prediction = model.predict(feature_vector_scaled)[0]
        prediction = max(0, int(round(prediction)))
        
        return prediction
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return 0

if __name__ == "__main__":
    # Test loading
    model, scaler, feature_names = load_trained_model()
    if model:
        print(f"Model loaded successfully with features: {feature_names}")