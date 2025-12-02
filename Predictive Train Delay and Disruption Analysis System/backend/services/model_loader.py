# a simple model loader / predict stub so predict endpoint works
# Replace with actual ML model loads (joblib / pickle / xgboost.load_model etc.)

def load_model():
    # If you have a real model, load it here and return
    return None

def load_scaler():
    return None

def load_model_predict_stub(features: dict):
    """
    Simple deterministic stub prediction using feature sum.
    Replace with: model.predict(some_feature_vector)
    """
    try:
        s = sum(float(v) for v in features.values())
        # example: predicted delay (minutes) = (sum mod 10) - 2  (just a synthetic number)
        predicted_delay = (s % 10) - 2
        return {"pred_minutes": float(predicted_delay)}
    except Exception:
        return {"pred_minutes": 0.0}