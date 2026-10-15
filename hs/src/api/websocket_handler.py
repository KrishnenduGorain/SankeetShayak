"""WebSocket Streaming Handler with Real-Time Sliding Buffer.

Authored for the API & Backend Workstream (Lead: Harsimran).
Maintains per-client circular frame buffers to evaluate kinematics, dynamics and sign predictions over live
WebSocket connections.
"""

from collections import deque
from typing import Deque, Dict, List, Optional
from fastapi import WebSocket
import numpy as np

from src.config import settings
from src.models.predictor import GesturePredictor
from src.vision.metric import MetricScaler


class StreamSession:
    """Manages an active real-time gesture streaming session."""

    def __init__(self, buffer_size: Optional[int] = None, fps: float = 30.0, predictor: Optional[GesturePredictor] = None,
                 min_frames: Optional[int] = None):
        self.buffer_size = buffer_size or settings.STREAM_BUFFER_SIZE
        self.min_frames = max(5, min_frames or settings.MIN_FRAMES_INFERENCE)
        self.fps = fps
        self.buffer: Deque[np.ndarray] = deque(maxlen=self.buffer_size)
        self.predictor = predictor or GesturePredictor(fps=fps)
        self.scaler = MetricScaler()
        self.frame_count = 0

    def push_frame(self, landmarks_21x3: np.ndarray, space: str = "metric", width: int = 1280, height: int = 720) -> Optional[Dict]:
        """Add a frame; returns a prediction dict once enough frames are buffered, else None.

        Raises ValueError for malformed input (caller reports it to the client).
        """
        lm = np.asarray(landmarks_21x3, dtype=np.float64)
        if lm.shape != (21, 3) or not np.all(np.isfinite(lm)):
            raise ValueError("landmarks must be a finite 21x3 array")
        if space == "image":
            metric = self.scaler.to_metric(lm, width, height)
            if metric is None:
                return None                       # hand too small / degenerate: skip frame
            lm = metric
        elif space != "metric":
            raise ValueError("space must be 'metric' or 'image'")

        self.buffer.append(lm)
        self.frame_count += 1
        if len(self.buffer) < self.min_frames:
            return None

        prediction = self.predictor.predict_trajectory(np.array(self.buffer, dtype=np.float64))
        return {
            "type": "inference_update",
            "frame_idx": self.frame_count,
            "prediction": prediction.label,
            "confidence": round(prediction.confidence, 4),
            "probabilities": {k: round(v, 4) for k, v in prediction.probabilities.items()},
            "telemetry": {k: round(v, 4) for k, v in prediction.telemetry.items()},
        }


class WebSocketStreamManager:
    """Coordinates active WebSocket connections."""

    def __init__(self, predictor_factory=None):
        self.active_connections: List[WebSocket] = []
        self._shared_predictor: Optional[GesturePredictor] = None
        self._factory = predictor_factory or GesturePredictor

    @property
    def predictor(self) -> GesturePredictor:
        if self._shared_predictor is None:
            self._shared_predictor = self._factory()
        return self._shared_predictor

    async def connect(self, websocket: WebSocket) -> StreamSession:
        await websocket.accept()
        self.active_connections.append(websocket)
        return StreamSession(predictor=self.predictor)

    def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
