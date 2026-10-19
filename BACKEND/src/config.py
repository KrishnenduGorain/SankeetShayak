"""Application Configuration Module for SankeetShayak.

Centralizes all environment variable settings with sensible defaults.
Supports CLIENT_HOST, CORS_ORIGINS, and dynamic Render port bindings.
"""

import os
from pathlib import Path
from typing import List
from dotenv import load_dotenv

# Resolve project root and load .env from BACKEND/ or root
BASE_DIR = Path(__file__).resolve().parent.parent
env_path = BASE_DIR / ".env"
if not env_path.is_file() and (BASE_DIR.parent / ".env").is_file():
    env_path = BASE_DIR.parent / ".env"

load_dotenv(env_path)


class Settings:
    """Application settings loaded from environment variables."""

    # Server & Network
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "production")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "info")

    # Client Host & CORS (Cross-Origin Resource Sharing)
    CLIENT_HOST: str = os.getenv("CLIENT_HOST", "*")
    raw_origins = os.getenv("CORS_ORIGINS") or os.getenv("CLIENT_HOST") or os.getenv("HANDSHAYAK_CORS_ORIGINS") or "*"
    CORS_ORIGINS: List[str] = [
        origin.strip()
        for origin in raw_origins.split(",")
        if origin.strip()
    ]
    FRONTEND_URL: str = os.getenv("FRONTEND_URL", "http://localhost:5173")

    # Model & Asset Paths
    MODEL_PATH: str = os.getenv("MODEL_PATH", "models/handshayak_model.joblib")
    MEDIAPIPE_TASK_PATH: str = os.getenv("MEDIAPIPE_TASK_PATH", "models/hand_landmarker.task")

    # Camera & Vision
    DEFAULT_CAMERA_INDEX: int = int(os.getenv("DEFAULT_CAMERA_INDEX", "0"))
    DEFAULT_FPS: float = float(os.getenv("DEFAULT_FPS", "30.0"))

    # Biomechanical Physics Engine Defaults
    HAND_MASS_KG: float = float(os.getenv("HAND_MASS_KG", "0.40"))
    SAVGOL_WINDOW: int = int(os.getenv("SAVGOL_WINDOW", "7"))
    SAVGOL_POLYORDER: int = int(os.getenv("SAVGOL_POLYORDER", "2"))

    # Streaming Buffer
    STREAM_BUFFER_SIZE: int = int(os.getenv("STREAM_BUFFER_SIZE", "30"))
    MIN_FRAMES_INFERENCE: int = int(os.getenv("MIN_FRAMES_INFERENCE", "15"))


settings = Settings()
