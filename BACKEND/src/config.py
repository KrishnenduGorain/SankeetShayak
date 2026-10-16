"""Application Configuration Module for SankeetShayak.

Centralizes all environment variable settings with sensible defaults.
"""

import os
from pathlib import Path
from typing import List
from dotenv import load_dotenv

# Resolve project root and load .env if present
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


class Settings:
    """Application settings loaded from environment variables."""

    # Server & Network
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "info")

    # CORS & Client Security
    CORS_ORIGINS: List[str] = [
        origin.strip()
        for origin in os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000,*").split(",")
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
