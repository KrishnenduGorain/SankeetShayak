"""Pydantic data schemas for REST API and WebSocket streams."""

from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field, model_validator

MAX_TRAJECTORY_FRAMES = 600


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.2.0"
    project: str = "HandShayak"


class SignClassListResponse(BaseModel):
    classes: List[str]
    count: int


class TrajectoryPredictRequest(BaseModel):
    """Time series of 21 hand landmarks per frame.

    ``space="metric"``: coordinates already in metres (recommended; physics-correct).
    ``space="image"``: MediaPipe normalised image coordinates; ``width``/``height`` are the frame size and
    the server converts to metres using the hand's own size.
    """
    trajectory: List[List[List[float]]] = Field(..., description="(T, 21, 3) landmark coordinates")
    fps: float = Field(default=30.0, ge=1.0, le=240.0)
    space: Literal["metric", "image"] = "metric"
    width: int = Field(default=1280, ge=1, le=16384)
    height: int = Field(default=720, ge=1, le=16384)

    @model_validator(mode="after")
    def _check_shape(self):
        t = self.trajectory
        if not 5 <= len(t) <= MAX_TRAJECTORY_FRAMES:
            raise ValueError(f"trajectory must have between 5 and {MAX_TRAJECTORY_FRAMES} frames")
        for i, frame in enumerate(t):
            if len(frame) != 21 or any(len(p) != 3 for p in frame):
                raise ValueError(f"frame {i} must contain exactly 21 [x, y, z] landmarks")
        return self


class TelemetryData(BaseModel):
    peak_speed_m_s: float
    mean_speed_m_s: float
    peak_accel_m_s2: float
    peak_jerk_m_s3: float
    peak_ke_joules: float
    total_energy_j_s: float
    peak_force_n: float


class PredictResponse(BaseModel):
    label: str
    confidence: float
    probabilities: Dict[str, float]
    telemetry: TelemetryData


class StreamFramePayload(BaseModel):
    frame_idx: int = 0
    timestamp_sec: Optional[float] = None
    landmarks: List[List[float]]
    space: Literal["metric", "image"] = "metric"
    width: int = 1280
    height: int = 720
