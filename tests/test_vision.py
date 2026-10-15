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


def test_detector_extract_from_video_shape(tmp_path):
    """Verify that video trajectory extraction returns array of shape (num_frames, 21, 3)."""
    import cv2
    video_path = tmp_path / "mock_video.avi"
    num_frames = 10
    w, h = 320, 240

    # Write a simple mock video file
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out = cv2.VideoWriter(str(video_path), fourcc, 30.0, (w, h))
    for _ in range(num_frames):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        out.write(frame)
    out.release()

    detector = HandLandmarkDetector()
    traj = detector.extract_from_video(video_path, max_frames=num_frames)

    assert isinstance(traj, np.ndarray)
    assert traj.shape == (num_frames, 21, 3)
    detector.close()


def test_cross_platform_camera_video_file(tmp_path):
    """Verify CrossPlatformCamera opens, reads, and yields frames from video."""
    import cv2
    from src.vision.camera import CrossPlatformCamera, get_platform_camera_backends, get_camera_permission_guidance

    backends = get_platform_camera_backends()
    assert len(backends) >= 1

    guidance = get_camera_permission_guidance()
    assert len(guidance) > 10

    video_path = tmp_path / "camera_test.avi"
    num_frames = 8
    w, h = 320, 240
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out = cv2.VideoWriter(str(video_path), fourcc, 30.0, (w, h))
    for _ in range(num_frames):
        out.write(np.zeros((h, w, 3), dtype=np.uint8))
    out.release()

    with CrossPlatformCamera(source=video_path) as cam:
        assert cam.is_opened()
        res_w, res_h = cam.get_resolution()
        assert res_w == w
        assert res_h == h
        frames = list(cam.frames(max_frames=num_frames))
        assert len(frames) == num_frames
        idx, first_frame = frames[0]
        assert idx == 0
        assert first_frame.shape == (h, w, 3)


def test_hand_visualizer_telemetry_hud_and_vectors():
    """Verify HandVisualizer renders velocity vectors and telemetry HUD correctly."""
    visualizer = HandVisualizer()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_lms = np.full((21, 3), 0.5, dtype=np.float32)
    mock_vel = np.ones((21, 3), dtype=np.float64) * 0.8

    # Test velocity vectors
    with_vectors = visualizer.draw_velocity_vectors(frame, mock_lms, mock_vel)
    assert with_vectors.shape == frame.shape
    assert np.any(with_vectors > 0)

    # Test HUD overlay
    telemetry = {
        "speed_m_s": 1.25,
        "accel_m_s2": 4.50,
        "jerk_m_s3": 35.0,
        "ke_joules": 0.125,
        "force_n": 1.80,
    }
    with_hud = visualizer.draw_telemetry_hud(
        with_vectors,
        telemetry=telemetry,
        prediction_label="TAP_FORCEFUL",
        confidence=0.98,
    )
    assert with_hud.shape == frame.shape
    assert np.any(with_hud > 0)


def test_stream_processor_on_video(tmp_path):
    """Verify StreamProcessor processes video frames and computes physics telemetry."""
    import cv2
    from src.vision.stream_processor import StreamProcessor

    video_path = tmp_path / "stream_proc_test.avi"
    num_frames = 12
    w, h = 320, 240
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out = cv2.VideoWriter(str(video_path), fourcc, 30.0, (w, h))
    for _ in range(num_frames):
        out.write(np.zeros((h, w, 3), dtype=np.uint8))
    out.release()

    with StreamProcessor(window_size=15, fps=30.0, enable_predictor=False) as processor:
        results = list(processor.process_source(source=video_path, display=False))
        assert len(results) == num_frames
        idx, annotated, telemetry = results[0]
        assert idx == 0
        assert annotated.shape == (h, w, 3)
        assert "speed_m_s" in telemetry
        assert "ke_joules" in telemetry


def test_opengl_renderer_offscreen():
    """Verify OpenGLHandRenderer renders 3D hand skeleton to offscreen buffer."""
    from src.vision.opengl_renderer import OpenGLHandRenderer, check_opengl_support

    support = check_opengl_support()
    assert "headless_capable" in support
    assert support["headless_capable"] is True

    renderer = OpenGLHandRenderer(width=320, height=240, elevation=20.0, azimuth=30.0)
    mock_lms = np.random.uniform(-0.2, 0.2, size=(21, 3)).astype(np.float64)
    mock_vel = np.random.uniform(-0.5, 0.5, size=(21, 3)).astype(np.float64)

    # Render without velocities
    frame = renderer.render_hand_3d(mock_lms, show_grid=True, show_labels=True)
    assert frame.shape == (240, 320, 3)
    assert np.any(frame != renderer.background_color)

    # Render with velocities
    frame_with_vel = renderer.render_hand_3d(mock_lms, velocities=mock_vel, show_grid=True)
    assert frame_with_vel.shape == (240, 320, 3)
    assert np.any(frame_with_vel != renderer.background_color)


def test_mediapipeline_on_video(tmp_path):
    """Verify MediaPipeline extracts landmarks, computes telemetry, and supports rendering."""
    import cv2
    from src.vision.mediapipeline import MediaPipeline, process_source

    video_path = tmp_path / "mediapipe_test.avi"
    num_frames = 10
    w, h = 320, 240
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out = cv2.VideoWriter(str(video_path), fourcc, 30.0, (w, h))
    for _ in range(num_frames):
        out.write(np.zeros((h, w, 3), dtype=np.uint8))
    out.release()

    pipeline = MediaPipeline(target_fps=30.0)
    assert pipeline is not None

    # Test process_source with rendering enabled
    lms_arr, telemetry, rendered_frames = process_source(
        source=str(video_path),
        max_frames=8,
        normalized=True,
        render=True,
        use_opengl=True,
    )

    assert isinstance(lms_arr, np.ndarray)
    assert lms_arr.shape == (8, 21, 3)
    assert "mean_speed" in telemetry
    assert "kinetic_energy" in telemetry
    assert telemetry["frame_count"] == 8
    assert rendered_frames is not None
    assert len(rendered_frames) == 8
    assert rendered_frames[0].shape == (480, 640, 3)
