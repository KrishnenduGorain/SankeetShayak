"""Detector -> metric scaling -> physics -> HUD, using the synthetic backend (no model, no camera)."""
import numpy as np
import pytest

from src.data.collector import GestureDatasetBuilder
from src.vision.camera import CrossPlatformCamera, get_camera_permission_guidance, probe_available_cameras
from src.vision.detector import (HandLandmarkDetector, ModelUnavailableError, SyntheticBackend, ensure_model)
from src.vision.metric import REF_PALM_LEN_M, MetricScaler
from src.vision.render import VideoSink, blank_canvas
from src.vision.stream_processor import StreamProcessor

W, H = 640, 480


def _tap_pair():
    ds = GestureDatasetBuilder.generate_synthetic_dataset(1, participants=1, seed=7)
    by = dict(zip(ds.labels, ds.trajectories))
    g, f = by["tap_gentle"].copy(), by["tap_forceful"].copy()
    f += g[-1, 0] - f[0, 0]                                 # continuous: no teleport between the taps
    return g, f


def test_metric_scaler_recovers_metres():
    scaler = MetricScaler()
    lm = np.zeros((21, 3)); lm[0] = [0.4, 0.5, 0]; lm[9] = [0.4, 0.5 - 100 / H, 0]       # palm = 100 px
    m = scaler.to_metric(lm, W, H)
    assert np.linalg.norm(m[0] - m[9]) == pytest.approx(REF_PALM_LEN_M, rel=1e-9)
    assert scaler.to_metric(np.zeros((21, 3)), W, H) is None                               # degenerate hand
    assert scaler.to_metric(np.full((21, 3), np.nan), W, H) is None


def test_metric_landmarks_keep_global_motion():
    """Regression for the original bug: wrist-normalised landmarks never move, metric ones do."""
    g, _ = _tap_pair()
    det = HandLandmarkDetector.with_backend(SyntheticBackend(g, W, H))
    first = det.process_frame(blank_canvas(W, H)); last = None
    for _ in range(len(g) - 1):
        last = det.process_frame(blank_canvas(W, H))
    assert np.allclose(first.normalized_landmarks[0], 0) and np.allclose(last.normalized_landmarks[0], 0)
    assert np.linalg.norm(last.metric_landmarks[0] - first.metric_landmarks[0]) > 0.05


def test_stream_processor_measures_gentle_vs_forceful():
    g, f = _tap_pair()
    traj = np.concatenate([g, f])
    sp = StreamProcessor(detector=HandLandmarkDetector.with_backend(SyntheticBackend(traj, W, H)), enable_predictor=False)
    peaks = [0.0, 0.0]
    for i in range(len(traj)):
        frame, tel = sp.process_frame(blank_canvas(W, H), timestamp_sec=i / 30.0)
        assert frame.shape == (H, W, 3) and np.isfinite(list(tel.values())).all()
        if i >= 6:
            peaks[0 if i < len(g) else 1] = max(peaks[0 if i < len(g) else 1], tel["speed_m_s"])
    assert 0.2 < peaks[0] < 0.6 and 0.8 < peaks[1] < 1.8 and peaks[1] > 2 * peaks[0]
    sp.close()


def test_stream_processor_with_predictor_and_hand_loss(model_path):
    from src.models.predictor import GesturePredictor
    g, _ = _tap_pair()
    sp = StreamProcessor(detector=HandLandmarkDetector.with_backend(SyntheticBackend(g, W, H)),
                         predictor=GesturePredictor(str(model_path)))
    for i in range(len(g)):
        sp.process_frame(blank_canvas(W, H), timestamp_sec=i / 30.0)
    assert sp.last_prediction is not None and len(sp.buffer) > 0
    for i in range(StreamProcessor.HAND_LOST_RESET_FRAMES + 1):         # backend exhausted -> no hand
        _, tel = sp.process_frame(blank_canvas(W, H), timestamp_sec=5 + i / 30.0)
    assert len(sp.buffer) == 0 and tel["speed_m_s"] == 0.0


def test_process_source_over_video_file_headless(tmp_path):
    g, _ = _tap_pair()
    src = tmp_path / "in.avi"
    with VideoSink(src, 30.0, (W, H), codecs=[("MJPG", ".avi")]) as sink:
        for _ in range(len(g)):
            sink.write(blank_canvas(W, H))
    sp = StreamProcessor(detector=HandLandmarkDetector.with_backend(SyntheticBackend(g, W, H)), enable_predictor=False)
    out = tmp_path / "out" / "annotated.mp4"
    n = sum(1 for _ in sp.process_source(str(src), display=True, save_output_path=out))     # display=True is safe headless
    assert n == len(g) and any(p.exists() for p in out.parent.glob("annotated.*"))


def test_extract_from_video_metric(tmp_path):
    g, _ = _tap_pair()
    src = tmp_path / "in.avi"
    with VideoSink(src, 30.0, (W, H), codecs=[("MJPG", ".avi")]) as sink:
        for _ in range(len(g)):
            sink.write(blank_canvas(W, H))
    det = HandLandmarkDetector.with_backend(SyntheticBackend(g, W, H))
    arr = det.extract_from_video(src, metric=True)
    assert arr.shape == (len(g), 21, 3) and np.abs(arr[-1, 0] - arr[0, 0]).max() > 0.05


def test_detector_input_validation_and_unknown_backend():
    det = HandLandmarkDetector.with_backend(SyntheticBackend(np.zeros((3, 21, 3)), W, H))
    with pytest.raises(ValueError):
        det.process_frame(np.zeros((10, 10), np.uint8))
    with pytest.raises(ValueError):
        HandLandmarkDetector(backend="nope")
    with pytest.raises(ValueError):
        SyntheticBackend(np.zeros((3, 20, 3)))


def test_model_download_disabled_gives_clear_error(tmp_path, monkeypatch):
    monkeypatch.setenv("HANDSHAYAK_OFFLINE", "1")
    with pytest.raises(ModelUnavailableError) as ei:
        ensure_model(tmp_path / "missing.task")
    assert "download_model.py" in str(ei.value)


def test_camera_helpers_are_safe_without_hardware(tmp_path):
    import os
    if not os.getenv("CI"):                      # opening a camera on CI macOS runners can block on a permission prompt
        assert isinstance(probe_available_cameras(1), list)
    assert get_camera_permission_guidance()
    with pytest.raises(FileNotFoundError):
        CrossPlatformCamera(source=str(tmp_path / "nope.mp4")).open()


@pytest.mark.model
def test_real_mediapipe_backend_blank_frame():
    """Runs only when the hand model is present (CI downloads it): no hand in a blank frame."""
    try:
        det = HandLandmarkDetector()
    except ModelUnavailableError:
        pytest.skip("hand_landmarker.task not available")
    try:
        res = det.process_frame(blank_canvas(W, H))
        assert res.detected is False
    finally:
        det.close()
