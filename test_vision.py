"""Unit tests for Vision Pipeline (src/vision)."""

import numpy as np
import pytest
from src.vision.detector import HandLandmarkDetector, LandmarkFrame
from src.vision.visualizer import HandVisualizer


def test_landmark_normalization_wrist_centered():
    """Verify wrist anchor (index 0) is translated to origin (0, 0, 0)."""
    raw_coords = np.zeros((21, 3), dtype=np.float32)
    raw_coords[0] = [10.0, 20.0, 30.0]  # Wrist offset
    raw_coords[9] = [10.0, 25.0, 30.0]  # Middle MCP offset (dist = 5.0)

    normalized = HandLandmarkDetector.normalize_coordinates(raw_coords)

    # Wrist must be at (0, 0, 0)
    np.testing.assert_allclose(normalized[0], [0.0, 0.0, 0.0], atol=1e-6)
    # Distance between wrist and Middle MCP must be exactly 1.0 (unit normalized)
    palm_len = np.linalg.norm(normalized[9])
    assert abs(palm_len - 1.0) < 1e-5


def test_hand_visualizer():
    """Verify HandVisualizer draws on frames without error."""
    visualizer = HandVisualizer()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Mock landmark points normalized [0, 1]
    mock_lms = np.random.uniform(0.1, 0.9, size=(21, 3)).astype(np.float32)
    annotated = visualizer.draw_skeleton(frame, mock_lms)

    assert annotated.shape == frame.shape
    # Ensure drawing modified the frame pixels
    assert np.any(annotated > 0)


def test_detector_blank_frame_no_detection():
    """Verify that a black frame gracefully produces detected=False without crashing."""
    detector = HandLandmarkDetector()
    blank_frame = np.zeros((240, 320, 3), dtype=np.uint8)

    result = detector.process_frame(blank_frame)

    assert isinstance(result, LandmarkFrame)
    assert not result.detected
    assert result.raw_landmarks is None
    detector.close()


# ---------------------------------------------------------------------------
# Extra vision tests (no camera or real hand needed)
# ---------------------------------------------------------------------------
import tempfile
from types import SimpleNamespace

from src.vision.capture import load_gesture, save_gesture
from src.vision.trajectory import fill_gaps, frames_to_arrays, longest_valid_run


def _fake_result(offset=0.0):
    """Build a MediaPipe-like result: a hand in a 640x480 frame."""
    rng = np.random.default_rng(0)
    pts = rng.uniform(0.3, 0.6, size=(21, 3))
    pts[:, 0] += offset
    world = rng.uniform(-0.05, 0.05, size=(21, 3))
    mk = lambda a: [SimpleNamespace(x=r[0], y=r[1], z=r[2]) for r in a]
    return SimpleNamespace(
        hand_landmarks=[mk(pts)],
        hand_world_landmarks=[mk(world)],
        handedness=[[SimpleNamespace(category_name="Right", score=0.9)]],
    )


def _detector_with_fake(results):
    det = HandLandmarkDetector()
    it = iter(results)
    calls = []

    def fake_detect(image, ts_ms):
        calls.append(ts_ms)
        return next(it)

    det.detector.detect_for_video = fake_detect
    det._calls = calls
    return det


def test_scale_invariance():
    rng = np.random.default_rng(1)
    coords = rng.uniform(0, 1, size=(21, 3))
    a = HandLandmarkDetector.normalize_coordinates(coords)
    b = HandLandmarkDetector.normalize_coordinates(coords * 3.0 + 7.0)
    np.testing.assert_allclose(a, b, atol=1e-5)


def test_zero_palm_length_does_not_crash():
    coords = np.zeros((21, 3))
    out = HandLandmarkDetector.normalize_coordinates(coords)
    assert np.all(np.isfinite(out))


def test_normalize_rejects_wrong_shape():
    with pytest.raises(ValueError):
        HandLandmarkDetector.normalize_coordinates(np.zeros((20, 3)))


def test_bad_frames_return_not_detected():
    det = HandLandmarkDetector()
    for bad in (None, np.zeros((0, 0, 3), dtype=np.uint8), np.zeros((4, 4, 2), dtype=np.uint8)):
        assert det.process_frame(bad).detected is False
    det.close()


def test_grayscale_and_bgra_frames_are_accepted():
    det = HandLandmarkDetector()
    assert det.process_frame(np.zeros((120, 160), dtype=np.uint8)).detected is False
    assert det.process_frame(np.zeros((120, 160, 4), dtype=np.uint8)).detected is False
    det.close()


def test_timestamps_strictly_increase_even_if_clock_repeats():
    det = _detector_with_fake([_fake_result() for _ in range(3)])
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for _ in range(3):
        det.process_frame(frame, timestamp_sec=1.0)  # same time 3 times
    assert det._calls == sorted(set(det._calls)) and len(det._calls) == 3
    det.close()


def test_detected_frame_has_all_outputs_and_metric_units():
    det = _detector_with_fake([_fake_result()])
    r = det.process_frame(np.zeros((480, 640, 3), dtype=np.uint8), 0.0)
    assert r.detected and r.handedness == "Right"
    for name in ("raw_landmarks", "world_landmarks", "normalized_landmarks", "metric_landmarks"):
        arr = getattr(r, name)
        assert arr.shape == (21, 3) and np.all(np.isfinite(arr)), name
    # wrist->middle MCP in metric space must equal the world palm length
    metric_palm = np.linalg.norm(r.metric_landmarks[9] - r.metric_landmarks[0])
    world_palm = np.linalg.norm(r.world_landmarks[9] - r.world_landmarks[0])
    assert abs(metric_palm - world_palm) < 1e-4
    det.close()


def test_metric_keeps_global_motion_but_normalized_does_not():
    det = _detector_with_fake([_fake_result(0.0), _fake_result(0.2)])
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    a = det.process_frame(frame, 0.0)
    b = det.process_frame(frame, 0.033)
    assert np.linalg.norm(b.metric_landmarks[0] - a.metric_landmarks[0]) > 1e-3
    np.testing.assert_allclose(a.normalized_landmarks[0], 0.0, atol=1e-6)
    np.testing.assert_allclose(b.normalized_landmarks[0], 0.0, atol=1e-6)
    det.close()


def test_aspect_ratio_does_not_distort_palm_scale():
    # Same physical palm vector, rotated 90 degrees, in a non-square frame
    px_a = np.zeros((21, 3)); px_a[9] = [100.0, 0.0, 0.0]
    px_b = np.zeros((21, 3)); px_b[9] = [0.0, 100.0, 0.0]
    na = HandLandmarkDetector.normalize_coordinates(px_a)
    nb = HandLandmarkDetector.normalize_coordinates(px_b)
    assert abs(np.linalg.norm(na[9]) - np.linalg.norm(nb[9])) < 1e-6


def test_fill_gaps_interpolates_short_and_keeps_long():
    traj = np.zeros((10, 21, 3))
    traj[:, :, 0] = np.arange(10)[:, None]
    valid = np.ones(10, dtype=bool)
    valid[2:4] = False          # short gap -> filled
    valid[6:10] = False         # trailing gap -> stays missing
    traj[~valid] = np.nan
    filled, v = fill_gaps(traj, valid, max_gap=3)
    assert v[2] and v[3] and not v[6]
    np.testing.assert_allclose(filled[2, 0, 0], 2.0)
    np.testing.assert_allclose(filled[3, 0, 0], 3.0)
    assert np.isnan(filled[7, 0, 0])
    assert longest_valid_run(v) == (0, 6)


def test_save_and_load_gesture_roundtrip():
    frames = []
    det = _detector_with_fake([_fake_result(0.01 * i) for i in range(8)])
    for i in range(8):
        frames.append(det.process_frame(np.zeros((480, 640, 3), dtype=np.uint8), i / 30.0))
    det.close()
    with tempfile.TemporaryDirectory() as d:
        out = save_gesture(f"{d}/tap_gentle_01", frames, "tap_gentle", subject="s1")
        g = load_gesture(str(out))
    assert g["label"] == "tap_gentle" and g["subject"] == "s1"
    assert g["metric"].shape == (8, 21, 3) and g["valid"].all()
    assert np.all(np.diff(g["timestamps"]) > 0)


def test_save_gesture_rejects_empty_recording():
    lost = [LandmarkFrame(frame_idx=i, timestamp_sec=i / 30.0, detected=False) for i in range(10)]
    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(ValueError):
            save_gesture(f"{d}/x.npz", lost, "x")


# ---------------------------------------------------------------------------
# Capture-loop tests with a pretend camera (no hardware needed)
# ---------------------------------------------------------------------------
import cv2
from src.vision import capture as capture_mod


class _FakeCap:
    def __init__(self, n_frames=5, opened=True):
        self.n = n_frames
        self.opened = opened
        self.released = False

    def isOpened(self):
        return self.opened and not self.released

    def read(self):
        if self.n <= 0:
            return False, None
        self.n -= 1
        return True, np.zeros((240, 320, 3), dtype=np.uint8)

    def release(self):
        self.released = True


def test_stream_landmarks_yields_frames_with_increasing_time(monkeypatch=None):
    fake = _FakeCap(5)
    orig = cv2.VideoCapture
    cv2.VideoCapture = lambda *a, **k: fake
    try:
        out = list(capture_mod.stream_landmarks(0))
    finally:
        cv2.VideoCapture = orig
    assert len(out) == 5
    times = [lf.timestamp_sec for lf, _ in out]
    assert all(b > a for a, b in zip(times, times[1:]))
    assert fake.released


def test_stream_landmarks_releases_camera_on_early_break():
    fake = _FakeCap(100)
    orig = cv2.VideoCapture
    cv2.VideoCapture = lambda *a, **k: fake
    try:
        gen = capture_mod.stream_landmarks(0)
        next(gen)
        gen.close()
    finally:
        cv2.VideoCapture = orig
    assert fake.released


def test_stream_landmarks_raises_when_camera_missing():
    fake = _FakeCap(opened=False)
    orig = cv2.VideoCapture
    cv2.VideoCapture = lambda *a, **k: fake
    try:
        with pytest.raises(RuntimeError):
            next(capture_mod.stream_landmarks(0))
    finally:
        cv2.VideoCapture = orig


def test_real_video_mode_runs_over_many_frames():
    """Real MediaPipe VIDEO mode (no fake): many frames, no timestamp errors."""
    det = HandLandmarkDetector()
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    for i in range(40):
        r = det.process_frame(frame, timestamp_sec=i / 30.0)
        assert r.detected is False
    # repeated / backwards timestamps must not crash either
    det.process_frame(frame, timestamp_sec=0.0)
    det.process_frame(frame, timestamp_sec=0.0)
    det.close()
    det.close()  # closing twice is safe
