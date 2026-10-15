import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run(args, model_path, tmp_path):
    env = {**os.environ, "MODEL_PATH": str(model_path), "HANDSHAYAK_HEADLESS": "1"}
    return subprocess.run([sys.executable, "demo.py", *args], cwd=ROOT, env=env, capture_output=True, text=True, timeout=240)


def test_demo_simulate(model_path, tmp_path):
    r = _run(["--simulate"], model_path, tmp_path)
    assert r.returncode == 0, r.stderr
    assert "Simulation complete" in r.stdout


def test_demo_selftest(model_path, tmp_path):
    r = _run(["--selftest"], model_path, tmp_path)
    assert r.returncode == 0 and "SELFTEST PASSED" in r.stdout, r.stdout + r.stderr


def test_demo_check_env(model_path, tmp_path):
    r = _run(["--check-env"], model_path, tmp_path)
    assert r.returncode == 0 and "OpenCV" in r.stdout, r.stdout + r.stderr


def test_demo_missing_video_is_a_clean_error(model_path, tmp_path):
    r = _run(["--video", str(tmp_path / "missing.mp4")], model_path, tmp_path)
    assert r.returncode == 1 and "Traceback" not in r.stderr, r.stderr
