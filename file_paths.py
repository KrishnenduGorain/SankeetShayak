"""Centralized file path configuration for SankeetShayak system."""

from pathlib import Path

# Base directory of the project
BASE_DIR = Path(__file__).parent

# Model files
MODEL_PATH = BASE_DIR / "models" / "handshayak_model.joblib"
MEDIAPIPE_TASK_PATH = BASE_DIR / "models" / "hand_landmarker.task"

# Data files
DATA_DIR = BASE_DIR / "data"
TRAINING_DATA_PATH = DATA_DIR / "training_data.npy"
TESTING_DATA_PATH = DATA_DIR / "testing_data.npy"

# Output files
OUTPUT_DIR = BASE_DIR / "output"
PREDICTIONS_PATH = OUTPUT_DIR / "predictions.json"
VISUALIZATIONS_PATH = OUTPUT_DIR / "visualizations"

# Configuration files
CONFIG_PATH = BASE_DIR / "config.py"
ENV_PATH = BASE_DIR / ".env"

# Ensure directories exist
for path in [BASE_DIR / "models", DATA_DIR, OUTPUT_DIR]:
    path.mkdir(parents=True, exist_ok=True)