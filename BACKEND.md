# SankeetShayak Backend Architecture & File Index

This file documents the entire backend architecture, directory structure, and runtime specifications for **SankeetShayak**.

---

## 📂 Project Directory Structure

```text
SankeetShayak/
├── BACKEND.code-workspace     # VS Code Workspace configuration for the Backend
├── BACKEND.md                 # Complete Backend Architecture & File Index
├── Dockerfile                 # Hardened multi-stage container (non-root appuser:1000)
├── docker-compose.yml         # Local development container orchestration
├── render.yaml                # 1-click Render Blueprint specification
├── requirements.txt           # Pinned production Python dependencies
├── pytest.ini                 # Pytest configuration (pythonpath & test discovery)
├── verify_backend.py          # Cross-platform pre-hosting diagnostic script
├── .env.example               # Environment variables template
├── .dockerignore              # Production container build exclusions
├── .gitignore                 # Version control exclusions
│
├── models/                    # Pre-trained models and weights
│   ├── hand_landmarker.task   # MediaPipe 3D Hand Landmark model binary (7.5 MB)
│   └── handshayak_model.joblib# Trained gesture classifier model
│
├── src/                       # Backend Source Code
│   ├── __init__.py            # Root package marker
│   ├── config.py              # Centralized environment settings with resilient fallbacks
│   │
│   ├── api/                   # FastAPI REST & WebSocket Service
│   │   ├── __init__.py
│   │   ├── main.py            # Application entrypoint, CORS & static frontend mount
│   │   ├── schemas.py         # Pydantic v2 data models with trajectory validation
│   │   └── websocket_handler.py # High-speed WebSocket streaming with worker threads
│   │
│   ├── models/                # Machine Learning & Gesture Inference
│   │   ├── __init__.py
│   │   ├── predictor.py       # GesturePredictor engine with input reliability checks
│   │   ├── train.py           # Model training pipeline
│   │   └── ablation.py        # Ablation study benchmarking (Shape vs Dynamics)
│   │
│   ├── physics/               # Biomechanical Numerical Physics
│   │   ├── __init__.py
│   │   ├── kinematics.py      # Savitzky-Golay smoothing (v, a, jerk derivatives)
│   │   └── dynamics.py        # Rigid-body mass segments (Kinetic energy, joint forces)
│   │
│   ├── vision/                # Computer Vision & 3D Graphics
│   │   ├── __init__.py
│   │   ├── camera.py          # Cross-platform camera capture (AVFoundation/MSMF/V4L2)
│   │   ├── detector.py        # MediaPipe 3D Landmark detection & normalization
│   │   ├── mediapipeline.py   # High-level pipeline tying camera to physics & models
│   │   ├── opengl_renderer.py # 3D OpenGL & offscreen perspective renderer
│   │   ├── visualizer.py      # 2D/3D skeleton drawing, velocity arrows, HUD
│   │   └── stream_processor.py# Stream orchestration processor
│   │
│   └── data/                  # Feature Extraction & Datasets
│       ├── __init__.py
│       ├── aggregator.py      # Fuses 72 shape + 15 dynamic features into 87D vector
│       └── collector.py       # Dataset collection and synthetic generator
│
├── tests/                     # Automated Test Suite (28 Passed)
│   ├── __init__.py
│   ├── test_api.py            # REST endpoints and WebSocket stream tests
│   ├── test_stream_session.py # 30-frame windowing, stride=3, NaN isolation
│   ├── test_models.py         # Model inference and reliability checks
│   ├── test_physics.py        # Numerical differentiation and dynamics tests
│   ├── test_vision.py         # MediaPipe normalization & offscreen 3D rendering
│   └── test_data.py           # Feature aggregator dimension tests
│
└── web/                       # Integrated Web Dashboard
    ├── index.html             # Client-side web dashboard
    ├── css/styles.css         # Dark-mode responsive styling
    └── js/                    # Client-side vision, physics, and telemetry
```

---

## 🚀 Running the Backend

### 1. In VS Code
- Open **`BACKEND.code-workspace`** via **File $\to$ Open Workspace from File...**
- Press **F5** or go to **Run and Debug** $\to$ Select **"Run FastAPI Backend (Uvicorn)"**.

### 2. From Terminal
```bash
# Activate environment
source venv/bin/activate

# Run verification diagnostic
python verify_backend.py

# Run all tests
pytest -q

# Start server
uvicorn src.api.main:app --reload --port 8000
```

---

## 🌐 API Endpoints Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the interactive Web Dashboard |
| `GET` | `/health` | Healthcheck returning `{"status": "ok"}` |
| `GET` | `/signs` | Returns supported sign classes from loaded model |
| `POST` | `/predict` | Evaluates trajectory, returns prediction + telemetry |
| `WebSocket` | `/ws/stream` | Real-time bi-directional streaming inference |
| `GET` | `/docs` | Interactive Swagger API documentation |
