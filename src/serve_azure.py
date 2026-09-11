from pathlib import Path
import json
import joblib
import numpy as np

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts"

MODEL_PATH = ARTIFACT_DIR / "ridge_model.joblib"
SCALE_PATH = ARTIFACT_DIR / "scale_params.json"
META_PATH = ARTIFACT_DIR / "model_metadata.json"


# Load production artifacts once when app starts
model = joblib.load(MODEL_PATH)

with open(SCALE_PATH) as f:
    scale_raw = json.load(f)

scale_raw.pop("_meta", None)
scale_params = scale_raw

with open(META_PATH) as f:
    model_meta = json.load(f)


app = FastAPI(
    title="Global Diabetes Prediction API",
    description="Predicts diabetes prevalence from obesity level and trend.",
    version="1.0.0",
)


class PredictionRequest(BaseModel):
    obesity_current: float = Field(..., ge=0, le=100)
    obesity_trend: float


class PredictionResponse(BaseModel):
    predicted_diabetes_prevalence_pct: float
    model_type: str
    test_r2: float


def scale_value(value: float, feature: str) -> float:
    params = scale_params[feature]
    return (value - params["mean"]) / params["std"]


@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "Global Diabetes Prediction API is running."
    }


@app.get("/health")
def health():
    return {
        "model_loaded": model is not None,
        "scale_params_loaded": scale_params is not None,
        "model_type": model_meta.get("model_type"),
        "model_version": model_meta.get("version"),
        "test_r2": model_meta.get("test_r2"),
        "cv_r2": model_meta.get("cv_r2"),
        "test_rmse": model_meta.get("test_rmse"),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        x = np.array([[
            scale_value(request.obesity_current, "obesity_level"),
            scale_value(request.obesity_trend, "obesity_trend"),
        ]])

        prediction = float(model.predict(x)[0])
        prediction = max(0.0, min(prediction, 50.0))

        return PredictionResponse(
            predicted_diabetes_prevalence_pct=round(prediction, 2),
            model_type=model_meta.get("model_type", "Ridge"),
            test_r2=model_meta.get("test_r2", 0.0),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
