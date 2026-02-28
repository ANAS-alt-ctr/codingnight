import os
import sys
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ml_project.api.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model_loaded" in data

def test_model_info():
    response = client.get("/model-info")
    # If model is loaded, we get 200, else 503
    assert response.status_code in [200, 503]
    if response.status_code == 200:
        data = response.json()
        assert "model_name" in data
        assert "features" in data

def test_predict_endpoint():
    payload = {
        "airline": "AA",
        "origin": "JFK",
        "dest": "LAX",
        "distance": 2475,
        "dep_delay": 20,
        "day_of_week": 1,
        "month": 6,
        "day_of_month": 15
    }
    response = client.post("/predict", json=payload)
    
    # If the model is not trained/loaded, we get 503
    if response.status_code == 503:
        pytest.skip("Model not loaded. Train model first before running full endpoint test.")
    
    assert response.status_code == 200
    data = response.json()
    assert "delay_probability" in data
    assert "prediction" in data
    assert data["prediction"] in ["Delayed", "On-Time"]
    assert "model_used" in data
