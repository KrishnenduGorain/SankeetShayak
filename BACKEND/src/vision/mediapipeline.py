"""MediaPipe End-to-End Hand Landmark Pipeline (Debonnita & Core Team).

Provides a unified cross-platform pipeline connecting:
1. Camera / Video capture (`CrossPlatformCamera`)
2. 3D Landmark extraction and normalization (`HandLandmarkDetector`)
3. Biomechanical kinematics smoothing & derivatives (`KinematicsEngine`)
4. Dynamic kinetic energy and joint torques (`DynamicsEngine`)
5. 2D & 3D OpenGL / OpenCV visualization (`HandVisualizer`, `OpenGLHandRenderer`)
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from src.physics.dynamics import DynamicsEngine
from src.physics.kinematics import KinematicsEngine
from src.vision.camera import CrossPlatformCamera
from src.vision.detector import HandLandmarkDetector, LandmarkFrame
from src.vision.opengl_renderer import OpenGLHandRenderer
from src.vision.visualizer import HandVisualizer

logger = logging.getLogger(__name__)


class MediaPipeline:
    """Unified MediaPipe 3D Landmark & Biomechanics Pipeline."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        min_hand_detection_confidence: float = 0.7,
        min_hand_presence_confidence: float = 0.7,
        min_tracking_confidence: float = 0.7,
        target_fps: float = 30.0,
    ):
        self.detector = HandLandmarkDetector(
            model_path=model_path,
            min_hand_detection_confidence=min_hand_detection_confidence,
            min_hand_presence_confidence=min_hand_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.kinematics_engine = KinematicsEngine(default_fps=target_fps)
        self.dynamics_engine = DynamicsEngine()
        self.visualizer = HandVisualizer()
        self.gl_renderer = OpenGLHandRenderer()
        self.target_fps = target_fps

    def process_frame(
        self,
        frame: np.ndarray,
        timestamp_sec: Optional[float] = None,
        normalized: bool = True,
    ) -> LandmarkFrame:
        """Process a single BGR image frame and return detected landmark frame."""
        return self.detector.detect_frame(frame, timestamp_sec=timestamp_sec)

    def run_on_capture(
        self,
        camera: CrossPlatformCamera,
        max_frames: Optional[int] = None,
        normalized: bool = True,
        render: bool = False,
        use_opengl: bool = False,
    ) -> Tuple[np.ndarray, Dict[str, Any], Optional[List[np.ndarray]]]:
        """Runs the pipeline over an active CrossPlatformCamera stream.
        
        Args:
            camera: Active CrossPlatformCamera instance.
            max_frames: Max frames to process (None for unlimited until stream ends).
            normalized: Return wrist-centered & scale-normalized coordinates.
            render: If True, renders visual frames with HUD overlay.
            use_opengl: If True, uses the 3D OpenGL / offscreen renderer for output frames.
            
        Returns:
            landmarks_arr: np.ndarray of shape (num_frames, 21, 3).
            telemetry: Dictionary containing mean speed, max acceleration, total kinetic energy.
            rendered_frames: List of rendered images (if render=True), else None.
        """
        raw_trajectories: List[np.ndarray] = []
        norm_trajectories: List[np.ndarray] = []
        rendered_images: Optional[List[np.ndarray]] = [] if render else None

        frame_count = 0
        fps = camera.fps if camera.fps > 0 else self.target_fps

        for raw_frame in camera.stream():
            if max_frames is not None and frame_count >= max_frames:
                break

            ts = frame_count / fps
            lm_frame = self.detector.detect_frame(raw_frame, timestamp_sec=ts)

            if lm_frame.detected and lm_frame.normalized_landmarks is not None:
                norm_trajectories.append(lm_frame.normalized_landmarks)
                raw_trajectories.append(
                    lm_frame.world_landmarks if lm_frame.world_landmarks is not None else lm_frame.raw_landmarks
                )
            else:
                # Hold previous or default zeros
                zero_lms = np.zeros((21, 3), dtype=np.float32)
                norm_trajectories.append(zero_lms)
                raw_trajectories.append(zero_lms)

            # Optional rendering
            if render:
                current_lms = norm_trajectories[-1]
                if use_opengl:
                    img = self.gl_renderer.render_hand_3d(current_lms)
                else:
                    img = self.visualizer.draw_landmarks(raw_frame, lm_frame)
                rendered_images.append(img)

            frame_count += 1

        selected_trajectories = norm_trajectories if normalized else raw_trajectories
        landmarks_arr = (
            np.array(selected_trajectories, dtype=np.float32)
            if selected_trajectories
            else np.empty((0, 21, 3), dtype=np.float32)
        )

        # Compute physics telemetry over extracted trajectory
        telemetry = self._compute_telemetry(landmarks_arr, fps=fps)

        return landmarks_arr, telemetry, rendered_images

    def _compute_telemetry(self, trajectory: np.ndarray, fps: float) -> Dict[str, Any]:
        """Compute kinematic and dynamic metrics for trajectory."""
        if len(trajectory) < 5:
            return {
                "mean_speed": 0.0,
                "peak_velocity": 0.0,
                "max_acceleration": 0.0,
                "mean_jerk": 0.0,
                "kinetic_energy": 0.0,
                "frame_count": len(trajectory),
            }

        k_engine = KinematicsEngine(default_fps=fps)
        kin_state = k_engine.compute_kinematics(trajectory)

        d_engine = DynamicsEngine()
        dyn_state = d_engine.compute_dynamics(kin_state)

        wrist_speed = kin_state.speed[:, 0]
        mean_speed = float(np.mean(wrist_speed))
        peak_velocity = float(np.max(wrist_speed))

        max_accel = float(np.max(kin_state.accel_mag[:, 0]))
        mean_jerk = float(np.mean(kin_state.jerk_mag[:, 0]))

        return {
            "mean_speed": round(mean_speed, 4),
            "peak_velocity": round(peak_velocity, 4),
            "max_acceleration": round(max_accel, 4),
            "mean_jerk": round(mean_jerk, 4),
            "kinetic_energy": round(float(np.mean(dyn_state.kinetic_energy)), 4),
            "frame_count": len(trajectory),
        }


def process_source(
    source: Union[int, str] = 0,
    max_frames: Optional[int] = None,
    normalized: bool = True,
    render: bool = False,
    use_opengl: bool = False,
    model_path: Optional[str] = None,
) -> Tuple[np.ndarray, Dict[str, Any], Optional[List[np.ndarray]]]:
    """High-level function to run the full MediaPipeline on a webcam or video file.
    
    Args:
        source: Camera index (int, e.g. 0) or path to video file (str).
        max_frames: Maximum frames to capture and process.
        normalized: Whether to return wrist-normalized coordinates.
        render: Whether to generate rendered frames with visual telemetry.
        use_opengl: Whether to use the 3D OpenGL offscreen renderer.
        model_path: Optional custom path to hand_landmarker.task model file.
        
    Returns:
        (landmarks_array, telemetry_dict, optional_rendered_frames)
    """
    pipeline = MediaPipeline(model_path=model_path)
    with CrossPlatformCamera(source=source) as cam:
        return pipeline.run_on_capture(
            camera=cam,
            max_frames=max_frames,
            normalized=normalized,
            render=render,
            use_opengl=use_opengl,
        )
