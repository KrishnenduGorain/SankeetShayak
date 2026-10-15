"""FastAPI Backend Application Entrypoint.

Authored for the ML & Backend Workstream (Lead: Harsimran).
Exposes REST endpoints and high-speed WebSocket streaming for real-time sign recognition.
"""

from typing import List
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

from src.api.schemas import (
    HealthResponse,
    PredictResponse,
    SignClassListResponse,
    TelemetryData,
    TrajectoryPredictRequest,
)
from src.api.websocket_handler import WebSocketStreamManager
from src.config import settings
from src.models.predictor import GesturePredictor

app = FastAPI(
    title="SankeetShayak API",
    description="Physics-Informed Sign-Language Recognition Backend Service",
    version="0.1.0",
)

# Enable CORS for frontend integration (Vercel, Netlify, Render, Localhost)
allow_all = "*" in settings.CORS_ORIGINS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if allow_all else settings.CORS_ORIGINS,
    allow_credentials=not allow_all,
    allow_methods=["*"],
    allow_headers=["*"],
)

stream_manager = WebSocketStreamManager()
_predictor: GesturePredictor = None


def get_predictor() -> GesturePredictor:
    global _predictor
    if _predictor is None:
        _predictor = GesturePredictor()
    return _predictor


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """System health verification."""
    return HealthResponse()


@app.get("/signs", response_model=SignClassListResponse, tags=["Classification"])
def list_sign_classes():
    """List all supported sign language classes."""
    predictor = get_predictor()
    classes = list(getattr(predictor.model, "classes_", ["tap_gentle", "tap_forceful", "swipe_slow", "swipe_rapid"]))
    return SignClassListResponse(classes=[str(c) for c in classes], count=len(classes))


@app.post("/predict", response_model=PredictResponse, tags=["Classification"])
def predict_gesture(request: TrajectoryPredictRequest):
    """Predict sign label and compute biomechanical metrics from a 3D coordinate trajectory."""
    predictor = get_predictor()
    traj_arr = np.array(request.trajectory, dtype=np.float64)

    result = predictor.predict_trajectory(traj_arr)

    return PredictResponse(
        label=result.label,
        confidence=result.confidence,
        probabilities=result.probabilities,
        telemetry=TelemetryData(**result.telemetry),
    )


@app.websocket("/ws/stream")
async def websocket_stream_endpoint(websocket: WebSocket):
    """Real-time bi-directional streaming endpoint.

    Expects JSON payload with 21 coordinates: {"landmarks": [[x, y, z], ...]}
    Streams back predictions and live telemetry.
    """
    session = await stream_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_json()
            if "landmarks" in data:
                lms = np.array(data["landmarks"], dtype=np.float32)
                if lms.shape == (21, 3):
                    update = session.push_frame(lms)
                    if update is not None:
                        await websocket.send_json(update)
    except WebSocketDisconnect:
        stream_manager.disconnect(websocket)
    except Exception as e:
        stream_manager.disconnect(websocket)


# Mount static web frontend if available (e.g. for Render or local web dashboard)
from pathlib import Path
from fastapi.staticfiles import StaticFiles

_web_dir = Path(__file__).resolve().parent.parent.parent / "web"
if _web_dir.is_dir() and (_web_dir / "index.html").is_file():
    app.mount("/", StaticFiles(directory=str(_web_dir), html=True), name="static")
