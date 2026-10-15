#!/usr/bin/env python3
"""Download the MediaPipe hand_landmarker.task model (atomic, retried)."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.vision.detector import ModelUnavailableError, default_model_path, ensure_model  # noqa: E402


def main() -> int:
    try:
        path = ensure_model(default_model_path())
    except ModelUnavailableError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Model ready: {path} ({path.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
