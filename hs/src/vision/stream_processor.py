"""Real-Time Stream Processor.

Connects Camera -> MediaPipe Vision -> Kinematics -> Dynamics -> Predictor -> Visualizer HUD.
Runs on macOS, Windows and Linux, headless or with a window.

Differences from the first version
* physics runs on METRIC landmarks (metres, moving with the hand) - the old wrist-centred buffer had zero
  global motion and palm-length units, so real speeds were ~0;
* the window is cleared after the hand is lost (no derivatives across a gap);
* a window is never opened without a usable display (``cv2.imshow`` would abort the process on headless Linux);
* frame timestamps drive the derivatives (webcam FPS is not constant).
"""

from collections import deque
import logging
from pathlib import Path
import time
from typing import Callable, Deque, Dict, Generator, Optional, Tuple, Union
import cv2
import numpy as np

from src.models.predictor import GesturePredictor
from src.physics.dynamics import PALM_INDICES, DynamicsEngine
from src.physics.kinematics import KinematicsEngine
from src.vision.camera import CrossPlatformCamera
from src.vision.detector import HandLandmarkDetector
from src.vision.render import SafeDisplay, VideoSink
from src.vision.visualizer import HandVisualizer

log = logging.getLogger(__name__)

EMPTY_TELEMETRY: Dict[str, float] = {"speed_m_s": 0.0, "accel_m_s2": 0.0, "jerk_m_s3": 0.0, "ke_joules": 0.0, "force_n": 0.0}


class StreamProcessor:
    """End-to-end real-time stream processor with physics engine & biomechanics."""

    HAND_LOST_RESET_FRAMES = 6

    def __init__(
        self,
        window_size: int = 30,
        fps: float = 30.0,
        enable_predictor: bool = True,
        detector: Optional[HandLandmarkDetector] = None,
        predictor: Optional[GesturePredictor] = None,
    ):
        """
        Args:
            window_size: frames in the rolling physics buffer.
            fps: nominal stream frame rate (real timestamps override it when provided).
            enable_predictor: run ML classification.
            detector / predictor: dependency injection (tests use a synthetic detector backend).
        """
        if window_size < 5:
            raise ValueError("window_size must be >= 5")
        self.window_size = window_size
        self.fps = fps
        self.dt = 1.0 / fps

        self.detector = detector if detector is not None else HandLandmarkDetector()
        self.visualizer = HandVisualizer()
        self.kinematics_engine = KinematicsEngine(default_fps=fps, savgol_window=7, savgol_polyorder=3)
        self.dynamics_engine = DynamicsEngine()
        self.predictor = predictor if predictor is not None else (GesturePredictor(fps=fps) if enable_predictor else None)

        self.buffer: Deque[np.ndarray] = deque(maxlen=window_size)       # metric (21, 3) landmarks
        self.time_buffer: Deque[float] = deque(maxlen=window_size)
        self._lost = 0
        self._t0 = time.perf_counter()
        self.last_prediction = None

    # ------------------------------------------------------------------ per frame
    def process_frame(self, frame_bgr: np.ndarray, timestamp_sec: Optional[float] = None) -> Tuple[np.ndarray, Dict]:
        """Run one frame through vision + physics. Returns (annotated_frame, telemetry)."""
        if timestamp_sec is None:
            timestamp_sec = time.perf_counter() - self._t0
        res = self.detector.process_frame(frame_bgr, timestamp_sec=timestamp_sec)
        annotated = frame_bgr.copy()
        telemetry: Dict[str, float] = dict(EMPTY_TELEMETRY)
        pred_label: Optional[str] = None
        pred_conf: Optional[float] = None

        if res.detected and res.metric_landmarks is not None and res.raw_landmarks is not None:
            self._lost = 0
            annotated = self.visualizer.draw_skeleton(annotated, res.raw_landmarks)
            if self.time_buffer and timestamp_sec <= self.time_buffer[-1]:
                timestamp_sec = self.time_buffer[-1] + 1e-3             # keep time strictly increasing
            self.buffer.append(res.metric_landmarks.astype(np.float64))
            self.time_buffer.append(timestamp_sec)

            if len(self.buffer) >= 5:
                traj = np.array(self.buffer, dtype=np.float64)
                kin = self.kinematics_engine.analyze_trajectory(traj, fps=self.fps, timestamps=np.array(self.time_buffer))
                dyn = self.dynamics_engine.compute_dynamics(kin)
                palm_v = kin.velocities[-1, list(PALM_INDICES), :].mean(axis=0)
                telemetry = {
                    "speed_m_s": float(np.linalg.norm(palm_v)),
                    "accel_m_s2": float(np.linalg.norm(kin.accelerations[-1, list(PALM_INDICES), :].mean(axis=0))),
                    "jerk_m_s3": float(np.linalg.norm(kin.jerks[-1, list(PALM_INDICES), :].mean(axis=0))),
                    "ke_joules": float(dyn.kinetic_energy[-1]),
                    "force_n": float(np.linalg.norm(dyn.inferred_force[-1])),
                }
                annotated = self.visualizer.draw_velocity_vectors(
                    annotated, res.raw_landmarks, kin.velocities[-1], meters_per_pixel=res.meters_per_pixel)
                if self.predictor is not None and len(self.buffer) >= 15:
                    pred = self.predictor.predict_trajectory(traj)
                    self.last_prediction = pred
                    pred_label, pred_conf = pred.label, pred.confidence
        else:
            self._lost += 1
            if self._lost >= self.HAND_LOST_RESET_FRAMES and self.buffer:
                self.buffer.clear()
                self.time_buffer.clear()

        annotated = self.visualizer.draw_telemetry_hud(annotated, telemetry=telemetry,
                                                       prediction_label=pred_label, confidence=pred_conf)
        return annotated, telemetry

    # ------------------------------------------------------------------ sources
    def process_source(
        self,
        source: Union[int, str, Path] = 0,
        max_frames: Optional[int] = None,
        display: bool = False,
        window_title: str = "HandShayak Live Telemetry",
        save_output_path: Optional[Union[str, Path]] = None,
        on_frame: Optional[Callable[[int, np.ndarray, Dict], None]] = None,
    ) -> Generator[Tuple[int, np.ndarray, Dict], None, None]:
        """Process a camera index or video file. Yields ``(frame_idx, annotated_frame, telemetry)``.

        ``display=True`` is honoured only when a GUI is actually usable; otherwise the stream runs headless
        (a warning is logged) instead of crashing.
        """
        sink: Optional[VideoSink] = None
        is_file = isinstance(source, (str, Path))
        window = SafeDisplay(window_title, enabled=display)
        try:
            with CrossPlatformCamera(source=source) as camera:
                fps = camera.get_fps()
                if is_file:
                    self.fps = fps
                    self.kinematics_engine = KinematicsEngine(default_fps=fps, savgol_window=7, savgol_polyorder=3)
                w, h = camera.get_resolution()
                if save_output_path is not None:
                    sink = VideoSink(save_output_path, fps, (w, h))
                t0 = time.perf_counter()
                for frame_idx, frame in camera.frames(max_frames=max_frames):
                    ts = frame_idx / fps if is_file else time.perf_counter() - t0   # file: media time; camera: wall clock
                    annotated, telemetry = self.process_frame(frame, timestamp_sec=ts)
                    if sink is not None:
                        sink.write(annotated)
                    if on_frame is not None:
                        on_frame(frame_idx, annotated, telemetry)
                    window.show(annotated)
                    if window.quit_requested():
                        break
                    yield frame_idx, annotated, telemetry
        finally:
            if sink is not None:
                sink.close()
            window.close()

    def close(self) -> None:
        if getattr(self, "detector", None) is not None:
            self.detector.close()

    def __enter__(self) -> "StreamProcessor":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
