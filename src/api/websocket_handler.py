"""WebSocket Streaming Handler with Non-Blocking Worker Thread Inference.

Provides robust per-client circular frame buffering, NaN/bad-frame isolation,
stride-based thread-pool inference, and explicit reset controls.
"""

import asyncio
from collections import deque
import logging
from typing import Deque, Dict, List, Optional
from fastapi import WebSocket, WebSocketDisconnect
import numpy as np

from src.models.predictor import GesturePredictor, get_shared_predictor

logger = logging.getLogger(__name__)


class StreamSession:
    """Manages an active real-time gesture streaming session with worker inference."""

    def __init__(
        self,
        buffer_size: int = 30,
        fps: float = 30.0,
        predictor: Optional[GesturePredictor] = None,
        stride: int = 3,
    ):
        self.buffer_size = buffer_size
        self.fps = fps
        self.stride = stride
        self.buffer: Deque[np.ndarray] = deque(maxlen=buffer_size)
        self.predictor = predictor or get_shared_predictor(allow_autobuild=True)
        self.frame_count = 0

    def reset(self):
        """Clear buffered frames and reset counters."""
        self.buffer.clear()
        self.frame_count = 0

    def push_frame(self, landmarks_21x3: np.ndarray) -> Tuple_Result:
        """Add a frame to the buffer synchronously.

        Returns:
            Tuple of (is_valid: bool, error_msg: Optional[str], should_infer: bool, traj_snapshot: Optional[np.ndarray])
        """
        try:
            lm = np.asarray(landmarks_21x3, dtype=np.float64)
        except Exception:
            return False, "Malformed landmark coordinates", False, None

        if lm.shape != (21, 3):
            return False, f"Expected landmark shape (21, 3), got {lm.shape}", False, None

        if not np.all(np.isfinite(lm)):
            return False, "Landmark coordinates contain NaN or Infinite values", False, None

        self.buffer.append(lm)
        self.frame_count += 1

        # Only run inference on a FULL 30-frame window and every `stride` frames
        is_full = len(self.buffer) >= self.buffer_size
        should_infer = is_full and (self.frame_count % self.stride == 0)

        snapshot = np.array(self.buffer, dtype=np.float64) if should_infer else None
        return True, None, should_infer, snapshot


Tuple_Result = tuple


class WebSocketStreamManager:
    """Coordinates active WebSocket connections with non-blocking worker inference."""

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> StreamSession:
        await websocket.accept()
        self.active_connections.append(websocket)
        predictor = get_shared_predictor(allow_autobuild=True)
        return StreamSession(predictor=predictor)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def handle_message(self, websocket: WebSocket, session: StreamSession, data: dict):
        """Handle incoming WebSocket payload safely."""
        # 1. Reset command
        if data.get("reset") is True:
            session.reset()
            await websocket.send_json({"type": "session_reset", "status": "buffer_cleared"})
            return

        # 2. Process landmarks
        if "landmarks" in data:
            valid, err_msg, should_infer, snapshot = session.push_frame(data["landmarks"])
            if not valid:
                # Send error response without dropping socket
                await websocket.send_json({
                    "type": "error",
                    "code": "bad_frame",
                    "message": err_msg,
                })
                return

            if should_infer and snapshot is not None:
                # Offload feature extraction and inference to worker thread
                prediction = await asyncio.to_thread(session.predictor.predict_trajectory, snapshot, session.fps)
                await websocket.send_json({
                    "type": "inference_update",
                    "frame_idx": session.frame_count,
                    "prediction": prediction.label,
                    "confidence": round(prediction.confidence, 4),
                    "probabilities": {k: round(v, 4) for k, v in prediction.probabilities.items()},
                    "telemetry": {k: round(v, 4) for k, v in prediction.telemetry.items()},
                    "is_reliable": prediction.is_reliable,
                    "warnings": prediction.warnings,
                })
