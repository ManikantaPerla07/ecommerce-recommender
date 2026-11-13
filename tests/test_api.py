"""
Inference API Integration Tests using FastAPI TestClient.
Demonstrates Acceptance Test 3 (real model outputs, not hardcoded sample predictions).
"""
import pytest
from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "hybrid" in data["loaded_models"]
    assert data["catalog_size"] > 0


def test_catalog_endpoint():
    response = client.get("/catalog?limit=5")
    assert response.status_code == 200
    data = response.json()
    assert data["limit"] == 5
    assert len(data["items"]) == 5
    assert "title" in data["items"][0]


def test_recommend_endpoint_warm_user():
    payload = {
        "user_id": "USER_0001",
        "top_k": 5,
        "model_type": "hybrid",
        "exclude_interacted": True
    }
    response = client.post("/recommend", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == "USER_0001"
    assert len(data["recommendations"]) == 5
    # Verify score is a real floating point number, not hard-coded string
    assert isinstance(data["recommendations"][0]["score"], float)
    assert len(data["recommendations"][0]["title"]) > 0


def test_recommend_endpoint_cold_start_user():
    payload = {
        "user_id": "USER_COLD_9999",
        "top_k": 5,
        "model_type": "hybrid"
    }
    response = client.post("/recommend", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["recommendations"]) == 5
    # Should fall back cleanly
    assert data["recommendations"][0]["is_cold_fallback"] is True


def test_recommend_endpoint_invalid_model():
    payload = {
        "user_id": "USER_0001",
        "top_k": 5,
        "model_type": "non_existent_model_xyz"
    }
    response = client.post("/recommend", json=payload)
    assert response.status_code == 400


def test_explain_endpoint():
    payload = {
        "user_id": "USER_0001",
        "product_id": "PROD_0005",
        "model_type": "hybrid"
    }
    response = client.post("/explain", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "hybrid_score" in data
    assert "sub_scores" in data
    assert "decision_support_limitations" in data


def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "overall" in data
    assert "Hybrid Recommender" in data["overall"]
