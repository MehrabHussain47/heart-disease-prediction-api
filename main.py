from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
from fastapi import FastAPI, HTTPException

from schemas import HeartFeatures, PredictionResponse

MODEL_PATH = Path("model/heart_model.joblib")
model: Any = None
metadata: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, metadata
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model not found at {MODEL_PATH}. Run `python train_model.py --data data/heart.csv`."
        )
    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    metadata = artifact["metadata"]
    yield


app = FastAPI(
    title="Heart Disease Prediction API",
    version="1.0.0",
    description="A small Docker-ready FastAPI service for binary heart-disease predictions.",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model_loaded": str(model is not None).lower()}


@app.get("/info")
def info() -> dict[str, Any]:
    return {
        "model_type": metadata.get("model_type", type(model).__name__),
        "features": metadata.get("features", []),
        "target_column": metadata.get("target_column", "target"),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(features: HeartFeatures) -> PredictionResponse:
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    row = [[getattr(features, name) for name in metadata["features"]]]
    prediction = int(model.predict(row)[0])
    probability = float(model.predict_proba(row)[0][1]) if hasattr(model, "predict_proba") else None
    return PredictionResponse(heart_disease=bool(prediction), probability=probability)
