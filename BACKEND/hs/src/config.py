"""Application Configuration Module for HandShayak.

Centralizes environment-variable settings with sensible defaults. ``python-dotenv`` is optional:
if it is not installed the app still starts (a ``.env`` file is simply not read).
"""

import os
from pathlib import Path
from typing import List

BASE_DIR = Path(__file__).resolve().parent.parent

try:  # optional dependency
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env")
except ImportError:  # pragma: no cover
    pass


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


class Settings:
    """Application settings loaded from environment variables."""

    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = _int("PORT", 8000)
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "info")

    # Wildcard origins are not combined with credentials (browsers reject that and it is unsafe).
    CORS_ORIGINS: List[str] = [
        o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000,http://localhost:5500").split(",")
        if o.strip()
    ]
    FRONTEND_URL: str = os.getenv("FRONTEND_URL", "http://localhost:5173")

    MODEL_PATH: str = os.getenv("MODEL_PATH", "models/handshayak_model.joblib")
    MEDIAPIPE_TASK_PATH: str = os.getenv("MEDIAPIPE_TASK_PATH", "models/hand_landmarker.task")

    DEFAULT_CAMERA_INDEX: int = _int("DEFAULT_CAMERA_INDEX", 0)
    DEFAULT_FPS: float = _float("DEFAULT_FPS", 30.0)

    HAND_MASS_KG: float = _float("HAND_MASS_KG", 0.40)
    SAVGOL_WINDOW: int = _int("SAVGOL_WINDOW", 7)
    SAVGOL_POLYORDER: int = _int("SAVGOL_POLYORDER", 3)     # 3 is required for a true jerk estimate

    STREAM_BUFFER_SIZE: int = _int("STREAM_BUFFER_SIZE", 30)
    MIN_FRAMES_INFERENCE: int = _int("MIN_FRAMES_INFERENCE", 15)

    @property
    def model_path(self) -> Path:
        p = Path(self.MODEL_PATH)
        return p if p.is_absolute() else BASE_DIR / p

    @property
    def mediapipe_task_path(self) -> Path:
        p = Path(self.MEDIAPIPE_TASK_PATH)
        return p if p.is_absolute() else BASE_DIR / p


settings = Settings()
