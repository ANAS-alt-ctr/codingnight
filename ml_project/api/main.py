"""
FastAPI Inference Endpoint for Flight Delay Prediction.
POST /predict: Returns delay probability and prediction.
GET /health: Health check.
GET /model-info: Returns best model info.
"""
import os
import sys
import pickle
import datetime
import traceback
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import BEST_MODEL_PATH

app = FastAPI(
    title="Flight Delay Prediction API",
    description="Predicts whether a flight will be delayed by more than 15 minutes.",
    version="1.0.0"
)

# ---- Load model at startup ----
MODEL_PAYLOAD = None


def load_model():
    global MODEL_PAYLOAD
    if os.path.exists(BEST_MODEL_PATH):
        with open(BEST_MODEL_PATH, "rb") as f:
            MODEL_PAYLOAD = pickle.load(f)
        print(f"[API] Loaded model: {MODEL_PAYLOAD['model_name']}")
    else:
        print(f"[API] WARNING: No model found at {BEST_MODEL_PATH}. Run training first.")


load_model()


# ---- Request / Response Schemas ----
class FlightInput(BaseModel):
    airline: str = Field(..., example="AA", description="IATA airline code")
    origin: str = Field(..., example="JFK", description="Origin airport")
    dest: str = Field(..., example="LAX", description="Destination airport")
    distance: float = Field(..., example=2475.0, description="Flight distance in miles")
    dep_delay: float = Field(0.0, example=15.0, description="Departure delay in minutes")
    day_of_week: int = Field(..., ge=0, le=6, example=1, description="Day of week (0=Mon, 6=Sun)")
    month: int = Field(..., ge=1, le=12, example=6, description="Month (1-12)")
    day_of_month: int = Field(1, ge=1, le=31, example=15, description="Day of month")


class PredictionResponse(BaseModel):
    delay_probability: float
    prediction: str
    model_used: str
    timestamp: str


# ---- Airline code mapping ----
KNOWN_AIRLINES = [
    "9E", "AA", "AS", "B6", "DL", "EV", "F9", "G4", "HA",
    "MQ", "NK", "OH", "OO", "QX", "UA", "VX", "WN", "YV", "YX"
]
AIRLINE_MAP = {a: i for i, a in enumerate(sorted(KNOWN_AIRLINES))}


def encode_input(inp: FlightInput, feature_cols: list) -> list:
    """Convert FlightInput to a feature vector matching training columns."""
    airline_code = AIRLINE_MAP.get(inp.airline.upper(), -1)
    is_weekend = 1 if inp.day_of_week in [5, 6] else 0

    if inp.distance <= 500:
        dist_bucket = 0
    elif inp.distance <= 1000:
        dist_bucket = 1
    elif inp.distance <= 2000:
        dist_bucket = 2
    else:
        dist_bucket = 3

    raw = {
        "DEP_DELAY": inp.dep_delay,
        "DISTANCE": inp.distance,
        "AIRLINE_CODE": airline_code,
        "DAY_OF_WEEK": inp.day_of_week,
        "MONTH": inp.month,
        "IS_WEEKEND": is_weekend,
        "DISTANCE_BUCKET": dist_bucket,
        "ORIGIN_FREQ": 0.05,   # approximation for unseen airports
        "DEST_FREQ": 0.05,
    }
    return [[raw.get(col, 0) for col in feature_cols]]
        

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": MODEL_PAYLOAD is not None,
        "timestamp": datetime.datetime.utcnow().isoformat()
    }


@app.get("/model-info")
def model_info():
    if MODEL_PAYLOAD is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run training first.")
    return {
        "model_name": MODEL_PAYLOAD["model_name"],
        "features": MODEL_PAYLOAD["feature_cols"],
        "metrics": MODEL_PAYLOAD["metrics"]
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(flight: FlightInput):
    if MODEL_PAYLOAD is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run training first.")

    try:
        feature_cols = MODEL_PAYLOAD["feature_cols"]
        model = MODEL_PAYLOAD["model"]
        X = encode_input(flight, feature_cols)

        prob = float(model.predict_proba(X)[0][1])
        label = "Delayed" if prob >= 0.5 else "On-Time"

        return PredictionResponse(
            delay_probability=round(prob, 4),
            prediction=label,
            model_used=MODEL_PAYLOAD["model_name"],
            timestamp=datetime.datetime.utcnow().isoformat()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {traceback.format_exc()}")


if __name__ == "__main__":
    import uvicorn
    from config import API_HOST, API_PORT
    uvicorn.run("ml_project.api.main:app", host=API_HOST, port=API_PORT, reload=True)
