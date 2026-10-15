"""FastAPI Backend Application Entrypoint.

Authored for the ML & Backend Workstream (Lead: Harsimran).
REST endpoints and WebSocket streaming for real-time sign recognition.

    uvicorn src.api.main:app --host 0.0.0.0 --port 8000
"""

import asyncio
import json
from contextlib import asynccontextmanager
import logging
import threading

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

from src.api.schemas import (
    HealthResponse, PredictResponse, SignClassListResponse, TelemetryData, TrajectoryPredictRequest,
)
from src.api.websocket_handler import WebSocketStreamManager
from src.config import settings
from src.data.collector import CLASS_NAMES
from src.models.predictor import GesturePredictor
from src.vision.metric import MetricScaler

log = logging.getLogger("handshayak.api")

_predictor = None
_predictor_lock = threading.Lock()


def get_predictor() -> GesturePredictor:
    """Thread-safe lazy singleton (training on first use may take a few seconds)."""
    global _predictor
    if _predictor is None:
        with _predictor_lock:
            if _predictor is None:
                _predictor = GesturePredictor()
    return _predictor


@asynccontextmanager
async def lifespan(app: FastAPI):
    await asyncio.to_thread(get_predictor)          # warm the model so the first request is fast
    yield


app = FastAPI(
    title="HandShayak API",
    description="Physics-Informed Sign-Language Recognition Backend Service",
    version="0.2.0",
    lifespan=lifespan,
)

_wildcard = "*" in settings.CORS_ORIGINS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if _wildcard else settings.CORS_ORIGINS,
    allow_credentials=not _wildcard,
    allow_methods=["*"],
    allow_headers=["*"],
)

stream_manager = WebSocketStreamManager(predictor_factory=get_predictor)


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """System health verification."""
    return HealthResponse()


@app.get("/signs", response_model=SignClassListResponse, tags=["Classification"])
def list_sign_classes():
    """List all supported sign classes."""
    classes = [str(c) for c in getattr(get_predictor().model, "classes_", CLASS_NAMES)]
    return SignClassListResponse(classes=classes, count=len(classes))


@app.post("/predict", response_model=PredictResponse, tags=["Classification"])
def predict_gesture(request: TrajectoryPredictRequest):
    """Predict the sign label and biomechanical metrics from a 3D landmark trajectory."""
    traj = np.array(request.trajectory, dtype=np.float64)
    if not np.all(np.isfinite(traj)):
        raise HTTPException(status_code=422, detail="trajectory contains NaN or inf")
    if request.space == "image":
        scaler = MetricScaler()
        frames = [scaler.to_metric(f, request.width, request.height) for f in traj]
        frames = [f for f in frames if f is not None]
        if len(frames) < 5:
            raise HTTPException(status_code=422, detail="hand too small or degenerate in image-space trajectory")
        traj = np.array(frames)
    try:
        result = get_predictor().predict_trajectory(traj, fps=request.fps)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return PredictResponse(label=result.label, confidence=result.confidence,
                           probabilities=result.probabilities, telemetry=TelemetryData(**result.telemetry))


@app.websocket("/ws/stream")
async def websocket_stream_endpoint(websocket: WebSocket):
    """Real-time bidirectional streaming.

    Client -> server JSON: ``{"landmarks": [[x, y, z] x 21], "space": "metric" | "image", "width": W, "height": H}``
    Server -> client JSON: ``inference_update`` messages, or ``{"type": "error", "detail": ...}`` for bad input.
    """
    session = await stream_manager.connect(websocket)
    try:
        while True:
            try:
                data = json.loads(await websocket.receive_text())
            except ValueError:
                await websocket.send_json({"type": "error", "detail": "invalid JSON"})
                continue
            if not isinstance(data, dict) or "landmarks" not in data:
                await websocket.send_json({"type": "error", "detail": "expected JSON object with 'landmarks'"})
                continue
            try:
                update = session.push_frame(np.array(data["landmarks"], dtype=np.float64), space=data.get("space", "metric"),
                                            width=int(data.get("width", 1280)), height=int(data.get("height", 720)))
            except (ValueError, TypeError) as exc:
                await websocket.send_json({"type": "error", "detail": str(exc)})
                continue
            if update is not None:
                await websocket.send_json(update)
    except WebSocketDisconnect:
        pass
    except Exception:  # unexpected: log, then close with an internal-error code
        log.exception("WebSocket stream failed")
        try:
            await websocket.close(code=1011)
        except RuntimeError:
            pass
    finally:
        stream_manager.disconnect(websocket)
