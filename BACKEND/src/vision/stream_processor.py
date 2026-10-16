"""Real-Time Stream Processor for SankeetShayak.

Connects Camera -> MediaPipe Vision -> Kinematics Engine -> Dynamics Engine ->
Predictor -> Visualizer HUD.
Runs cross-platform on macOS, Windows, and Linux.
"""

from collections import deque
from pathlib import Path
import sys
import time
from typing import Callable, Dict, Generator, Optional, Tuple, Union
import cv2
import numpy as np

from src.models.predictor import GesturePredictor
from src.physics.dynamics import DynamicsEngine
from src.physics.kinematics import KinematicsEngine
from src.vision.camera import CrossPlatformCamera
from src.vision.detector import HandLandmarkDetector
from src.vision.visualizer import HandVisualizer


class StreamProcessor:
    """End-to-end real-time stream processor with physics engine & biomechanics."""

    def __init__(
        self,
        window_size: int = 30,
        fps: float = 30.0,
        enable_predictor: bool = True,
    ):
        """Initialize StreamProcessor pipeline.

        Args:
            window_size: Number of frames in rolling buffer for physics smoothing.
            fps: Video stream frame rate.
            enable_predictor: Whether to execute ML gesture classification.
        """
        self.window_size = window_size
        self.fps = fps
        self.dt = 1.0 / fps

        # Initialize pipeline modules
        self.detector = HandLandmarkDetector()
        self.visualizer = HandVisualizer()
        self.kinematics_engine = KinematicsEngine(default_fps=fps, savgol_window=7, savgol_polyorder=2)
        self.dynamics_engine = DynamicsEngine()

        self.predictor = GesturePredictor(fps=fps) if enable_predictor else None

        # Rolling landmark buffer: stores normalized (21, 3) landmarks
        self.buffer: deque = deque(maxlen=window_size)
        self.raw_buffer: deque = deque(maxlen=window_size)

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        timestamp_sec: Optional[float] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Process a single frame through the complete vision & physics pipeline.

        Args:
            frame_bgr: OpenCV image array (H, W, 3).
            timestamp_sec: Optional frame timestamp in seconds.

        Returns:
            Tuple of (annotated_frame_bgr, telemetry_dict).
        """
        res = self.detector.process_frame(frame_bgr, timestamp_sec=timestamp_sec)
        annotated = frame_bgr.copy()

        telemetry: Dict[str, float] = {
            "speed_m_s": 0.0,
            "accel_m_s2": 0.0,
            "jerk_m_s3": 0.0,
            "ke_joules": 0.0,
            "force_n": 0.0,
        }
        pred_label: Optional[str] = None
        pred_conf: Optional[float] = None

        if res.detected and res.raw_landmarks is not None and res.normalized_landmarks is not None:
            # 1. Draw hand skeleton
            annotated = self.visualizer.draw_skeleton(annotated, res.raw_landmarks)

            # 2. Append coordinates to rolling buffer
            self.buffer.append(res.normalized_landmarks)
            self.raw_buffer.append(res.raw_landmarks)

            # 3. Compute real-time kinematics and dynamics if buffer has enough frames
            if len(self.buffer) >= 3:
                traj_arr = np.array(self.buffer, dtype=np.float64)
                kin_state = self.kinematics_engine.analyze_trajectory(traj_arr, fps=self.fps)
                dyn_state = self.dynamics_engine.compute_dynamics(kin_state)

                # Instantaneous metrics (latest frame)
                curr_speed = float(np.mean(kin_state.speed[-1]))
                curr_accel = float(np.mean(kin_state.accel_mag[-1]))
                curr_jerk = float(np.mean(kin_state.jerk_mag[-1]))
                curr_ke = float(dyn_state.kinetic_energy[-1])
                curr_force = float(np.linalg.norm(dyn_state.inferred_force[-1]))

                telemetry = {
                    "speed_m_s": curr_speed,
                    "accel_m_s2": curr_accel,
                    "jerk_m_s3": curr_jerk,
                    "ke_joules": curr_ke,
                    "force_n": curr_force,
                }

                # 4. Draw fingertip velocity arrows
                latest_vel = kin_state.velocities[-1]
                annotated = self.visualizer.draw_velocity_vectors(
                    annotated, res.raw_landmarks, latest_vel, scale=25.0
                )

                # 5. Predict gesture label if window is warm
                if self.predictor is not None and len(self.buffer) >= 15:
                    pred = self.predictor.predict_trajectory(traj_arr)
                    pred_label = pred.label
                    pred_conf = pred.confidence

        # 6. Render HUD overlay
        annotated = self.visualizer.draw_telemetry_hud(
            annotated,
            telemetry=telemetry,
            prediction_label=pred_label,
            confidence=pred_conf,
        )

        return annotated, telemetry

    def process_source(
        self,
        source: Union[int, str, Path] = 0,
        max_frames: Optional[int] = None,
        display: bool = False,
        window_title: str = "SankeetShayak Live Telemetry",
        save_output_path: Optional[Union[str, Path]] = None,
        on_frame: Optional[Callable[[int, np.ndarray, Dict], None]] = None,
    ) -> Generator[Tuple[int, np.ndarray, Dict], None, None]:
        """Run continuous stream processing over camera or video file.

        Args:
            source: Camera index (e.g. 0) or path to video file (.mp4, .avi).
            max_frames: Optional frame limit.
            display: Whether to open an OpenCV display window (press 'q' to stop).
            window_title: Title for cv2.imshow window.
            save_output_path: Optional path to save annotated video.
            on_frame: Optional callback invoked on each processed frame.

        Yields:
            (frame_idx, annotated_frame, telemetry_dict)
        """
        video_writer = None

        with CrossPlatformCamera(source=source) as camera:
            fps = camera.get_fps()
            w, h = camera.get_resolution()

            if save_output_path is not None:
                save_str = str(save_output_path)
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(save_str, fourcc, fps, (w, h))

            can_display = display
            # Check for headless environment
            if can_display:
                if sys.platform.startswith("linux") and not (
                    "DISPLAY" in sys.argv or "WAYLAND_DISPLAY" in sys.argv
                ):
                    import os
                    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
                        can_display = False

            try:
                for frame_idx, frame in camera.frames(max_frames=max_frames):
                    annotated_frame, telemetry = self.process_frame(frame)

                    if video_writer is not None:
                        video_writer.write(annotated_frame)

                    if on_frame is not None:
                        on_frame(frame_idx, annotated_frame, telemetry)

                    if can_display:
                        cv2.imshow(window_title, annotated_frame)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break

                    yield frame_idx, annotated_frame, telemetry

            finally:
                if video_writer is not None:
                    video_writer.release()
                if can_display:
                    cv2.destroyAllWindows()

    def close(self):
        """Release underlying detector resources."""
        if hasattr(self, "detector") and self.detector:
            self.detector.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
