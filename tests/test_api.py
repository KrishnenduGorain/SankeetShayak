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
    """Verify /predict accepts trajectory and returns label + telemetry + reliability."""
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
    assert "is_reliable" in data
    assert "warnings" in data
    assert "peak_speed_m_s" in data["telemetry"]
    assert "peak_ke_joules" in data["telemetry"]


def test_predict_endpoint_validation_error():
    """Verify /predict returns 422 for malformed or short trajectory."""
    # Too short (< 15 frames)
    short_traj = np.zeros((10, 21, 3)).tolist()
    res = client.post("/predict", json={"trajectory": short_traj})
    assert res.status_code == 422

    # Bad fps (< 1.0)
    res_fps = client.post("/predict", json={"trajectory": np.zeros((20, 21, 3)).tolist(), "fps": -5.0})
    assert res_fps.status_code == 422

    # Bad coordinate dimensions (only 2 coords instead of 3)
    bad_dim = np.zeros((20, 21, 2)).tolist()
    res_dim = client.post("/predict", json={"trajectory": bad_dim})
    assert res_dim.status_code == 422


def test_websocket_stream_endpoint():
    """Verify WebSocket /ws/stream accepts frames and streams predictions upon full 30-frame window."""
    with client.websocket_connect("/ws/stream") as websocket:
        # Send 30 frames to fill the full buffer window
        for i in range(30):
            mock_landmarks = np.full((21, 3), 0.02 * i, dtype=np.float32).tolist()
            websocket.send_json({"landmarks": mock_landmarks})

        # Receive inference update from server
        data = websocket.receive_json()
        assert data["type"] == "inference_update"
        assert "prediction" in data
        assert "confidence" in data
        assert "telemetry" in data
        assert "is_reliable" in data

        # Send bad frame -> receives error without dropping
        websocket.send_json({"landmarks": [[float("nan")] * 3] * 21})
        err_data = websocket.receive_json()
        assert err_data["type"] == "error"
        assert err_data["code"] == "bad_frame"

        # Send reset command
        websocket.send_json({"reset": True})
        rst_data = websocket.receive_json()
        assert rst_data["type"] == "session_reset"
