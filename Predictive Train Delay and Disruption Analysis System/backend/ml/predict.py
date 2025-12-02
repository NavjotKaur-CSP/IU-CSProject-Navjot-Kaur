# ml/predict.py
import pandas as pd
import numpy as np
import shap
from services.model_loader import load_model, load_scaler
from utils.preprocess import simple_feature_engineer, scale_features
import xgboost as xgb

def prepare_input(row_dict):
    df = pd.DataFrame([row_dict])
    df["scheduled_time"] = pd.to_datetime(df["scheduled_time"])
    X = simple_feature_engineer(df)
    X_num = X.select_dtypes(include=[np.number])
    Xs = scale_features(X_num, fit_if_needed=False)
    return Xs, X_num

def predict_and_explain(row_dict):
    model_booster = load_model()
    if model_booster is None:
        raise RuntimeError("Model not found. Train the model first.")
    Xs, X_num = prepare_input(row_dict)
    dmatrix = xgb.DMatrix(Xs)
    pred = model_booster.predict(dmatrix)[0]
    # SHAP: need sklearn wrapper for explainer compatibility
    # Create a temporary XGBRegressor and load booster
    reg = xgb.XGBRegressor()
    reg._Booster = model_booster
    explainer = shap.Explainer(reg)
    shap_values = explainer(X_num)
    # return prediction and top contributing features
    top = sorted(zip(X_num.columns, shap_values.values[0]), key=lambda x: abs(x[1]), reverse=True)[:5]
    return {"prediction": float(pred), "top_features": [{ "feature": f, "shap": float(val)} for f,val in top]}
