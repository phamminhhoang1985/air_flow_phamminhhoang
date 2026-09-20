"""
FastAPI Serving Application: Load model from MLflow Registry and serve predictions.
"""
import os
from contextlib import asynccontextmanager

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_NAME = os.getenv("MODEL_NAME", "breast-cancer-classifier")
MODEL_VERSION = os.getenv("MODEL_VERSION", "1")
MODEL_URI = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")

model = None
load_error: str | None = None


def get_model():
    """Lazily load or reload model from MLflow Registry if not yet loaded."""
    global model, load_error
    if model is None:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        try:
            model = mlflow.sklearn.load_model(MODEL_URI)
            load_error = None
            print(f"Successfully loaded {MODEL_URI} from MLflow Registry")
        except Exception as exc:
            load_error = f"{type(exc).__name__}: {exc}"
    return model


@asynccontextmanager
async def lifespan(_: FastAPI):
    get_model()
    yield


app = FastAPI(title="DDM501 Tutorial 03 — Serving from MLflow Registry", lifespan=lifespan)


class PredictRequest(BaseModel):
    features: list[float] = Field(..., min_length=30, max_length=30)


@app.get("/health")
def health():
    active_model = get_model()
    if active_model is None:
        return {
            "status": "degraded",
            "model_loaded": False,
            "model_uri": MODEL_URI,
            "error": load_error,
        }
    return {"status": "ok", "model_loaded": True, "model_uri": MODEL_URI}


@app.post("/predict")
def predict(req: PredictRequest):
    active_model = get_model()
    if active_model is None:
        raise HTTPException(status_code=503, detail=f"{MODEL_URI} not loaded: {load_error}")
    
    # Feature names validation & column alignment
    columns = getattr(active_model, "feature_names_in_", None)
    if columns is not None:
        row = pd.DataFrame([req.features], columns=columns)
    else:
        row = pd.DataFrame([req.features])

    proba = float(active_model.predict_proba(row)[0][1])
    return {
        "prediction": "benign" if proba >= 0.5 else "malignant",
        "probability_benign": round(proba, 4),
        "served_by": MODEL_URI,
    }
