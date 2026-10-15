#!/usr/bin/env python3
"""Cross-Platform Backend Diagnostic & Pre-Hosting Verification Script.

Tests environment, models, physics, vision, REST API, WebSockets, and Render readiness.
Works on macOS, Linux, and Windows.
"""

import os
import platform
import sys
from pathlib import Path
import numpy as np

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"


def log_pass(msg: str):
    print(f"[{GREEN}PASS{RESET}] {msg}")


def log_fail(msg: str):
    print(f"[{RED}FAIL{RESET}] {msg}")


def log_info(msg: str):
    print(f"[{YELLOW}INFO{RESET}] {msg}")


def main() -> int:
    print("=" * 60)
    print(" SankeetShayak Backend Cross-Platform Diagnostic ")
    print("=" * 60)
    log_info(f"OS: {platform.system()} {platform.release()} ({platform.machine()})")
    log_info(f"Python: {platform.python_version()} ({sys.executable})")

    failures = 0

    # 1. Verify Model Assets
    print("\n--- 1. Model Assets Verification ---")
    models_dir = Path("models")
    task_model = models_dir / "hand_landmarker.task"
    clf_model = models_dir / "handshayak_model.joblib"

    if task_model.is_file() and task_model.stat().st_size > 1_000_000:
        log_pass(f"MediaPipe task model found: {task_model} ({task_model.stat().st_size / (1024*1024):.1f} MB)")
    else:
        log_fail(f"MediaPipe model missing or corrupted at {task_model}")
        failures += 1

    if clf_model.is_file() and clf_model.stat().st_size > 10_000:
        log_pass(f"Trained gesture model found: {clf_model} ({clf_model.stat().st_size / 1024:.1f} KB)")
    else:
        log_fail(f"Trained gesture model missing at {clf_model}")
        failures += 1

    # 2. Verify Physics Engines
    print("\n--- 2. Biomechanical Physics Verification ---")
    try:
        from src.physics.kinematics import KinematicsEngine
        from src.physics.dynamics import DynamicsEngine

        kin_engine = KinematicsEngine(default_fps=30.0)
        dyn_engine = DynamicsEngine()

        # Simulate 20 frames of harmonic motion
        t = np.linspace(0, 1.0, 20)
        mock_traj = np.zeros((20, 21, 3))
        mock_traj[:, :, 0] = np.sin(2 * np.pi * 2.0 * t)[:, None]

        kin_state = kin_engine.compute_kinematics(mock_traj)
        dyn_state = dyn_engine.compute_dynamics(kin_state)

        assert kin_state.velocities.shape == (20, 21, 3)
        assert len(dyn_state.kinetic_energy) == 20
        log_pass("Kinematics (Savitzky-Golay) and Dynamics (KE & Force) engines operational")
    except Exception as e:
        log_fail(f"Physics engine verification failed: {e}")
        failures += 1

    # 3. Verify Vision & OpenGL Rendering
    print("\n--- 3. Vision & OpenGL Offscreen Rendering ---")
    try:
        from src.vision.opengl_renderer import OpenGLHandRenderer, check_opengl_support

        support = check_opengl_support()
        log_pass(f"Graphics runtime detected: PyOpenGL={support['pyopengl']}, Headless={support['headless_capable']}")

        renderer = OpenGLHandRenderer(width=320, height=240)
        lms = np.random.uniform(-0.1, 0.1, size=(21, 3))
        frame = renderer.render_hand_3d(lms)
        assert frame.shape == (240, 320, 3)
        log_pass("3D Hand & joint offscreen renderer buffer generated successfully")
    except Exception as e:
        log_fail(f"Vision/OpenGL rendering failed: {e}")
        failures += 1

    # 4. Verify FastAPI Backend Endpoints
    print("\n--- 4. FastAPI REST Endpoints Verification ---")
    try:
        from fastapi.testclient import TestClient
        from src.api.main import app

        client = TestClient(app)

        # GET /health
        res_health = client.get("/health")
        assert res_health.status_code == 200 and res_health.json().get("status") == "ok"
        log_pass("REST GET /health: 200 OK")

        # GET /
        res_root = client.get("/")
        assert res_root.status_code == 200 and "HandShayak" in res_root.text
        log_pass("REST GET /: 200 OK (Web Dashboard served)")

        # GET /signs
        res_signs = client.get("/signs")
        assert res_signs.status_code == 200 and len(res_signs.json().get("classes", [])) > 0
        log_pass(f"REST GET /signs: 200 OK ({len(res_signs.json()['classes'])} classes)")

        # POST /predict
        sample_traj = np.random.normal(0, 0.05, size=(20, 21, 3)).tolist()
        res_pred = client.post("/predict", json={"trajectory": sample_traj, "fps": 30.0})
        assert res_pred.status_code == 200 and "label" in res_pred.json()
        log_pass(f"REST POST /predict: 200 OK (Label: {res_pred.json()['label']}, Conf: {res_pred.json()['confidence']:.2f})")
    except Exception as e:
        log_fail(f"FastAPI REST endpoints failed: {e}")
        failures += 1

    # 5. Verify Real-Time WebSocket Streaming
    print("\n--- 5. WebSocket Real-Time Stream Verification ---")
    try:
        with client.websocket_connect("/ws/stream") as ws:
            for i in range(16):
                pt = np.full((21, 3), 0.02 * i, dtype=np.float32).tolist()
                ws.send_json({"landmarks": pt})
            data = ws.receive_json()
            assert data.get("type") == "inference_update"
            assert "prediction" in data
            log_pass(f"WebSocket /ws/stream: streaming prediction received ({data['prediction']})")
    except Exception as e:
        log_fail(f"WebSocket stream verification failed: {e}")
        failures += 1

    # Summary
    print("\n" + "=" * 60)
    if failures == 0:
        print(f"{GREEN}ALL DIAGNOSTIC CHECKS PASSED (100% HEALTHY). READY FOR RENDER!{RESET}")
        print("=" * 60)
        return 0
    else:
        print(f"{RED}DIAGNOSTIC CHECKS FAILED: {failures} ERROR(S) FOUND.{RESET}")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
