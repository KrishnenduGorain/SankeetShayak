"""Unit tests for FastAPI REST and Streaming endpoints (src/api)."""

import numpy as np
import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_health_check_endpoint():
    """Verify /health returns HTTP 200 with status ok."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["project"] == "SankeetShayak"


def test_root_serves_frontend():
    """Verify root / serves HTML dashboard."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "HandShayak" in response.text


def test_signs_list_endpoint():
    """Verify /signs returns the list of recognizable gestures."""
    response = client.get("/signs")
    assert response.status_code == 200
    data = response.json()
    assert "classes" in data
    assert data["count"] > 0


def test_predict_endpoint_success():
    """Verify /predict accepts trajectory and returns label + telemetry."""
    # Create 20 frames of 21 3D points
    mock_traj = np.random.normal(0, 0.05, size=(20, 21, 3)).tolist()

    response = client.post(
        "/predict",
        json={"trajectory": mock_traj, "fps": 30.0}
    )

    assert response.status_code == 200
    data = response.json()
    assert "label" in data
    assert "confidence" in data
    assert "telemetry" in data
    assert "peak_speed_m_s" in data["telemetry"]
    assert "peak_ke_joules" in data["telemetry"]


def test_websocket_stream_endpoint():
    """Verify WebSocket /ws/stream accepts frames and streams predictions."""
    with client.websocket_connect("/ws/stream") as websocket:
        # Send 16 frames of 21 landmarks to fill the warmup buffer (minimum 15 frames)
        for i in range(16):
            mock_landmarks = np.full((21, 3), 0.05 * i, dtype=np.float32).tolist()
            websocket.send_json({"landmarks": mock_landmarks})

        # Receive inference update from server
        data = websocket.receive_json()
        assert data["type"] == "inference_update"
        assert "prediction" in data
        assert "confidence" in data
        assert "telemetry" in data

