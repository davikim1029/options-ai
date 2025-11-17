# evaluation.py
import pandas as pd
import joblib
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

MODEL_PATH = Path("models/versions/current_model.pkl")
SCALER_PATH = Path("models/versions/current_scaler.pkl")

EVAL_FEATURES = [
    "optionType","strikePrice","lastPrice","bid","ask","bidSize","askSize",
    "volume","openInterest","nearPrice","inTheMoney","delta","gamma","theta",
    "vega","rho","iv","spread","midPrice","moneyness","daysToExpiration"
]

TARGET_COLUMNS = ["predicted_return", "predicted_hold_days"]

def load_model_and_scaler():
    if not MODEL_PATH.exists():
        raise RuntimeError("Model not trained yet.")
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

def evaluate_model(df: pd.DataFrame) -> dict:
    df = df.dropna(subset=EVAL_FEATURES + TARGET_COLUMNS)
    model, scaler = load_model_and_scaler()

    X = df[EVAL_FEATURES].astype(float)
    y_true = df[TARGET_COLUMNS].astype(float)

    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)

    results = {}
    for i, target in enumerate(TARGET_COLUMNS):
        y_t = y_true[target]
        y_p = y_pred[:, i]

        results[target] = {
            "rmse": mean_squared_error(y_t, y_p, squared=False),
            "mae": mean_absolute_error(y_t, y_p),
            "r2": r2_score(y_t, y_p),
            "error_mean": float(np.mean(y_p - y_t)),
            "error_std": float(np.std(y_p - y_t))
        }

    return {
        "rows_evaluated": len(df),
        "metrics": results
    }
