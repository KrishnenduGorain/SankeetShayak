"""FastAPI Backend Application Entrypoint.

Provides hardened REST endpoints and non-blocking WebSocket streaming
for real-time physics-informed sign language recognition.
"""

import os
from pathlib import Path
from typing import List
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
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
from src.models.predictor import GesturePredictor, ModelNotFoundError, get_shared_predictor

app = FastAPI(
    title="SankeetShayak API",
    description="Physics-Informed Sign-Language Recognition Backend Service",
    version="0.1.0",
)

# CORS: Check explicit CLIENT_HOST, CORS_ORIGINS, or HANDSHAYAK_CORS_ORIGINS
env_client = os.getenv("CLIENT_HOST")
env_cors = os.getenv("CORS_ORIGINS") or os.getenv("HANDSHAYAK_CORS_ORIGINS") or env_client
allowed_origins = [o.strip() for o in env_cors.split(",") if o.strip()] if env_cors else settings.CORS_ORIGINS

allow_all = "*" in allowed_origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if allow_all else allowed_origins,
    allow_credentials=False if allow_all else True,
    allow_methods=["*"],
    allow_headers=["*"],
)

stream_manager = WebSocketStreamManager()


def get_predictor() -> GesturePredictor:
    """Retrieve shared predictor or raise HTTP 503 if model not found."""
    try:
        return get_shared_predictor(allow_autobuild=True)
    except ModelNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Inference model unavailable: {e}",
        )


@app.get("/health", response_model=HealthResponse, tags=["System"])
@app.head("/health", include_in_schema=False, tags=["System"])
def health_check():
    """System health verification."""
    return HealthResponse()


@app.get("/signs", response_model=SignClassListResponse, tags=["Classification"])
def list_sign_classes():
    """List supported sign classes directly from the loaded model."""
    predictor = get_predictor()
    classes = predictor.classes
    if not classes:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is loaded but contains no gesture classes.",
        )
    return SignClassListResponse(classes=classes, count=len(classes))


@app.post("/predict", response_model=PredictResponse, tags=["Classification"])
def predict_gesture(request: TrajectoryPredictRequest):
    """Predict sign label and compute biomechanical metrics from a 3D coordinate trajectory."""
    predictor = get_predictor()
    traj_arr = np.array(request.trajectory, dtype=np.float64)

    try:
        result = predictor.predict_trajectory(traj_arr, fps=request.fps)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))

    return PredictResponse(
        label=result.label,
        confidence=result.confidence,
        probabilities=result.probabilities,
        telemetry=TelemetryData(**result.telemetry),
        is_reliable=result.is_reliable,
        warnings=result.warnings,
    )


@app.websocket("/ws/stream")
async def websocket_stream_endpoint(websocket: WebSocket):
    """Real-time bi-directional streaming endpoint with non-blocking worker inference."""
    session = await stream_manager.connect(websocket)
    try:
        while True:
            try:
                data = await websocket.receive_json()
            except (ValueError, TypeError):
                await websocket.send_json({"type": "error", "message": "Invalid JSON payload"})
                continue
            await stream_manager.handle_message(websocket, session, data)
    except WebSocketDisconnect:
        stream_manager.disconnect(websocket)
    except Exception as exc:
        stream_manager.disconnect(websocket)


# Mount static web frontend if available (e.g. for Render or local web dashboard)
_base = Path(__file__).resolve().parent.parent.parent
_candidate_dirs = [
    _base / "web",
    _base / "FRONTEND",
    _base.parent / "FRONTEND",
]
_web_dir = next((d for d in _candidate_dirs if d.is_dir() and (d / "index.html").is_file()), None)
if _web_dir:
    app.mount("/", StaticFiles(directory=str(_web_dir), html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=settings.HOST,
        port=settings.PORT,
        log_level=settings.LOG_LEVEL.lower(),
        reload=(settings.ENVIRONMENT == "development"),
    )
