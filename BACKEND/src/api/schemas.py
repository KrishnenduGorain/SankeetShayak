"""Pydantic data schemas for REST API and WebSocket streams."""

from typing import Dict, List, Optional
import math
from pydantic import BaseModel, Field, field_validator


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"
    project: str = "SankeetShayak"


class SignClassListResponse(BaseModel):
    classes: List[str]
    count: int


class TrajectoryPredictRequest(BaseModel):
    # Shape: (T, 21, 3) normalized hand landmark coordinates
    trajectory: List[List[List[float]]] = Field(
        ...,
        description="Time series of 3D hand landmark coordinates (T, 21, 3)"
    )
    fps: float = Field(default=30.0, ge=1.0, le=120.0)

    @field_validator("trajectory")
    @classmethod
    def validate_trajectory(cls, v: List[List[List[float]]]) -> List[List[List[float]]]:
        if len(v) < 15:
            raise ValueError(f"Trajectory must have at least 15 frames for physical analysis, got {len(v)}")
        for i, frame in enumerate(v):
            if len(frame) != 21:
                raise ValueError(f"Frame {i} must have exactly 21 landmarks, got {len(frame)}")
            for j, pt in enumerate(frame):
                if len(pt) != 3:
                    raise ValueError(f"Landmark {j} in frame {i} must have [x, y, z] coordinates, got {len(pt)}")
                if any(math.isnan(coord) or math.isinf(coord) for coord in pt):
                    raise ValueError(f"NaN or Inf detected at frame {i}, joint {j}")
        return v


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
    is_reliable: bool = True
    warnings: List[str] = Field(default_factory=list)


class StreamFramePayload(BaseModel):
    frame_idx: int
    timestamp_sec: Optional[float] = None
    # 21 landmarks [x, y, z]
    landmarks: List[List[float]]
