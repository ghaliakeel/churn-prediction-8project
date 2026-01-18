"""Tests for FastAPI endpoints."""
import pytest
from fastapi.testclient import TestClient


def test_health_endpoint():
    """Test health check endpoint."""
    from churn_prediction.api import app

    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_root_endpoint():
    """Test root endpoint."""
    from churn_prediction.api import app

    client = TestClient(app)
    response = client.get("/")

    assert response.status_code == 200
    assert "status" in response.json()
