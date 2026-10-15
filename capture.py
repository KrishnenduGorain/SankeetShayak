"""Webcam capture, FPS measurement and gesture recording for the vision pipeline."""

import time
from pathlib import Path
from typing import Iterator, List, Optional, Tuple
import cv2
import numpy as np

from src.vision.detector import HandLandmarkDetector, LandmarkFrame
from src.vision.trajectory import fill_gaps, frames_to_arrays, longest_valid_run


def stream_landmarks(
    camera_index: int = 0,
    mirror: bool = True,
    max_frames: Optional[int] = None,
    detector: Optional[HandLandmarkDetector] = None,
) -> Iterator[Tuple[LandmarkFrame, np.ndarray]]:
    """Yield ``(LandmarkFrame, frame_bgr)`` from a webcam with real timestamps.

    The camera and detector are always released, even if the consumer stops
    iterating early (use it in a ``for`` loop and ``break`` freely).

    Raises:
        RuntimeError: if the camera cannot be opened.
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open camera {camera_index}")
    own_detector = detector is None
    det = detector or HandLandmarkDetector()
    t0 = time.perf_counter()
    n = 0
    try:
        while max_frames is None or n < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if mirror:
                frame = cv2.flip(frame, 1)
            lf = det.process_frame(frame, time.perf_counter() - t0)
            n += 1
            yield lf, frame
    finally:
        cap.release()
        if own_detector:
            det.close()


def measure_fps(camera_index: int = 0, seconds: float = 10.0) -> float:
    """Measure end-to-end pipeline FPS (capture + MediaPipe). Milestone 1 needs >= 30."""
    start = time.perf_counter()
    frames = 0
    for _lf, _ in stream_landmarks(camera_index):
        frames += 1
        if time.perf_counter() - start >= seconds:
            break
    elapsed = time.perf_counter() - start
    return frames / elapsed if elapsed > 0 else 0.0


def save_gesture(
    path: str,
    frames: List[LandmarkFrame],
    label: str,
    subject: str = "unknown",
    notes: str = "",
    max_gap: int = 3,
) -> Path:
    """Save one recorded gesture to a compressed .npz file.

    Stored arrays: ``metric`` (T,21,3) meters, ``normalized`` (T,21,3),
    ``timestamps`` (T,), ``valid`` (T,) bool, plus ``label``, ``subject``,
    ``notes``. Short gaps are interpolated; the longest valid run is what
    should be fed to the physics engine.

    Raises:
        ValueError: if too few frames contain a hand.
    """
    metric, ts, valid = frames_to_arrays(frames, "metric_landmarks")
    norm, _, valid_n = frames_to_arrays(frames, "normalized_landmarks")
    valid = valid & valid_n
    metric, valid_f = fill_gaps(metric, valid, max_gap)
    norm, _ = fill_gaps(norm, valid, max_gap)
    run = longest_valid_run(valid_f)
    if run is None or (run[1] - run[0]) < 5:
        raise ValueError("Recording has fewer than 5 consecutive frames with a hand")

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix != ".npz":
        out = out.with_suffix(".npz")
    np.savez_compressed(
        out,
        metric=metric, normalized=norm, timestamps=ts, valid=valid_f,
        label=np.array(label), subject=np.array(subject), notes=np.array(notes),
    )
    return out


def load_gesture(path: str) -> dict:
    """Load a gesture saved by :func:`save_gesture` (returns plain Python/NumPy values)."""
    with np.load(path, allow_pickle=False) as d:
        out = {k: d[k] for k in ("metric", "normalized", "timestamps", "valid")}
        for k in ("label", "subject", "notes"):
            out[k] = str(d[k])
    return out
