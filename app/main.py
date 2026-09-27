from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "xgboost_fraud_model.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
FEATURE_INFO_PATH = MODEL_DIR / "feature_info.pkl"


# ---------------------------------------------------------
# Load deployment artifacts
# ---------------------------------------------------------

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
feature_info = joblib.load(FEATURE_INFO_PATH)

FEATURES = feature_info["features"]


# ---------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------

app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="XGBoost API for detecting potentially fraudulent credit card transactions.",
    version="1.0.0"
)


# ---------------------------------------------------------
# Input schema
# ---------------------------------------------------------

class Transaction(BaseModel):
    Time: float = Field(..., description="Seconds elapsed since first transaction")
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float = Field(..., ge=0)


# ---------------------------------------------------------
# Routes
# ---------------------------------------------------------

@app.get("/")
def root():
    return {
        "message": "Credit Card Fraud Detection API",
        "status": "running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "XGBoost",
        "features": len(FEATURES)
    }


@app.post("/predict")
def predict(transaction: Transaction):

    try:
        # Convert request to dataframe
        input_data = pd.DataFrame(
            [transaction.model_dump()]
        )

        # Ensure training feature order
        input_data = input_data[FEATURES]

        # Scale only Time and Amount
        input_data[["Time", "Amount"]] = scaler.transform(
            input_data[["Time", "Amount"]]
        )

        # Prediction
        prediction = int(model.predict(input_data)[0])

        probability = float(
            model.predict_proba(input_data)[0][1]
        )

        return {
            "prediction": prediction,
            "label": "Fraud" if prediction == 1 else "Legitimate",
            "fraud_probability": round(probability, 6)
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc)
        )