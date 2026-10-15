#!/usr/bin/env python3
"""Cross-Platform Backend Diagnostic & Pre-Hosting Verification Script.

Uses Python's built-in pathlib module for OS-independent path resolution.
Verifies project structure, dependencies, models, physics, vision, REST API, and WebSockets.
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


def print_status(message: str, status: bool):
    symbol = f"{GREEN}[OK]{RESET}" if status else f"{RED}[FAIL]{RESET}"
    print(f"{symbol} {message}")


def check_project_structure() -> bool:
    print("\n--- 1. Running Cross-Platform Structure Verification ---")
    root = Path(__file__).resolve().parent

    # 1. Essential Directories Check
    src_exists = (root / "src").is_dir()
    models_exists = (root / "models").is_dir()
    tests_exists = (root / "tests").is_dir()

    print_status("Directory 'src/' exists", src_exists)
    print_status("Directory 'models/' exists", models_exists)
    print_status("Directory 'tests/' exists", tests_exists)

    # 2. Essential Files Check
    req_exists = (root / "requirements.txt").is_file()
    dockerfile_exists = (root / "Dockerfile").is_file()
    env_example_exists = (root / ".env.example").is_file()

    print_status("File 'requirements.txt' exists", req_exists)
    print_status("File 'Dockerfile' exists", dockerfile_exists)
    print_status("File '.env.example' exists", env_example_exists)

    # 3. Requirements Content Check (Server verification)
    server_found = False
    if req_exists:
        content = (root / "requirements.txt").read_text(encoding="utf-8").lower()
        if "uvicorn" in content or "gunicorn" in content:
            server_found = True
    print_status("Production ASGI/WSGI server (Uvicorn/Gunicorn) in requirements.txt", server_found)

    return all([src_exists, models_exists, tests_exists, req_exists, dockerfile_exists, env_example_exists, server_found])


def check_runtime_and_endpoints() -> bool:
    root = Path(__file__).resolve().parent
    failures = 0

    # 4. Model Assets Check
    print("\n--- 2. Model Assets Check ---")
    task_model = root / "models" / "hand_landmarker.task"
    clf_model = root / "models" / "handshayak_model.joblib"
    task_ok = task_model.is_file() and task_model.stat().st_size > 1_000_000
    clf_ok = clf_model.is_file() and clf_model.stat().st_size > 10_000
    print_status(f"MediaPipe task model ({task_model.name})", task_ok)
    print_status(f"Trained gesture model ({clf_model.name})", clf_ok)
    if not (task_ok and clf_ok):
        failures += 1

    # 5. Physics Engine Check
    print("\n--- 3. Biomechanical Physics Engine Check ---")
    try:
        from src.physics.kinematics import KinematicsEngine
        from src.physics.dynamics import DynamicsEngine

        kin = KinematicsEngine(default_fps=30.0)
        dyn = DynamicsEngine()
        traj = np.zeros((20, 21, 3))
        t = np.linspace(0, 1.0, 20)
        traj[:, :, 0] = np.sin(2 * np.pi * t)[:, None]
        kin_state = kin.compute_kinematics(traj)
        dyn_state = dyn.compute_dynamics(kin_state)
        print_status("Kinematics & Dynamics calculations", kin_state.velocities.shape == (20, 21, 3))
    except Exception as e:
        print_status(f"Physics engine: {e}", False)
        failures += 1

    # 6. Vision & OpenGL Offscreen Rendering
    print("\n--- 4. Vision & OpenGL Offscreen Check ---")
    try:
        from src.vision.opengl_renderer import OpenGLHandRenderer
        renderer = OpenGLHandRenderer(width=320, height=240)
        buf = renderer.render_hand_3d(np.zeros((21, 3)))
        print_status("3D Offscreen hand render buffer (320x240x3)", buf.shape == (240, 320, 3))
    except Exception as e:
        print_status(f"Vision/OpenGL rendering: {e}", False)
        failures += 1

    # 7. FastAPI REST & WebSocket Endpoints
    print("\n--- 5. FastAPI Endpoints Check ---")
    try:
        from fastapi.testclient import TestClient
        from src.api.main import app
        client = TestClient(app)

        res_h = client.get("/health")
        print_status("REST GET /health", res_h.status_code == 200 and res_h.json().get("status") == "ok")

        res_root = client.get("/")
        print_status("REST GET / (Web Dashboard)", res_root.status_code == 200 and "HandShayak" in res_root.text)

        res_signs = client.get("/signs")
        print_status(f"REST GET /signs ({len(res_signs.json().get('classes', []))} classes)", res_signs.status_code == 200)

        mock_traj = np.random.normal(0, 0.05, size=(20, 21, 3)).tolist()
        res_pred = client.post("/predict", json={"trajectory": mock_traj, "fps": 30.0})
        print_status(f"REST POST /predict (Label: {res_pred.json().get('label')})", res_pred.status_code == 200)

        with client.websocket_connect("/ws/stream") as ws:
            for i in range(30):
                ws.send_json({"landmarks": np.zeros((21, 3)).tolist()})
            msg = ws.receive_json()
            print_status(f"WebSocket /ws/stream (Prediction: {msg.get('prediction')})", msg.get("type") == "inference_update")
    except Exception as e:
        print_status(f"FastAPI endpoints: {e}", False)
        failures += 1

    return failures == 0


def main():
    print("=" * 55)
    print(" SankeetShayak Cross-Platform Backend Verification ")
    print("=" * 55)
    print(f"{YELLOW}Platform:{RESET} {platform.system()} {platform.machine()} | Python {platform.python_version()}")

    struct_ok = check_project_structure()
    runtime_ok = check_runtime_and_endpoints()

    print("\n---------------------------------------------------")
    if struct_ok and runtime_ok:
        print(f"{GREEN}Success: Core backend structure and runtime are valid and ready for Render!{RESET}")
        print("---------------------------------------------------\n")
        return 0
    else:
        print(f"{RED}Notice: Some critical checks failed. Review items marked [FAIL].{RESET}")
        print("---------------------------------------------------\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
