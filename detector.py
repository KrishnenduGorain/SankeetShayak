"""Hand Landmark Extraction using MediaPipe Tasks Vision.

Authored for the Vision Pipeline (Lead: Debonnita).
Provides high-performance, robust 3D hand tracking (21 keypoints)
with relative coordinate normalization (wrist anchor and palm scale).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# Landmark index definitions for reference
HAND_LANDMARK_NAMES = [
    "WRIST",  # 0
    "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",  # 1-4
    "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",  # 5-8
    "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",  # 9-12
    "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",  # 13-16
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP"  # 17-20
]

# Standard hand joint linkages for visualization & connectivity
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (0, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (5, 9), (9, 13), (13, 17), (0, 17)     # Palm base
]


@dataclass
class LandmarkFrame:
    """Represents a single frame's detected hand landmarks."""
    frame_idx: int
    timestamp_sec: float
    detected: bool
    # Raw coordinates (21, 3): [x, y, z] normalized to image coordinates
    raw_landmarks: Optional[np.ndarray] = None
    # World coordinates (21, 3) in meters, origin at hand center
    world_landmarks: Optional[np.ndarray] = None
    # Scale- and translation-invariant normalized coordinates (21, 3).
    # Good for SHAPE features (joint angles, finger distances). The wrist is
    # always at the origin, so it carries NO whole-hand motion.
    normalized_landmarks: Optional[np.ndarray] = None
    # Approximate metric coordinates (21, 3) in METERS that KEEP whole-hand
    # position in the image. Use this for velocity / acceleration / jerk.
    metric_landmarks: Optional[np.ndarray] = None
    handedness: str = "Unknown"  # "Left" or "Right"
    confidence: float = 0.0


class HandLandmarkDetector:
    """Detects 21 3D hand landmarks in video frames using MediaPipe HandLandmarker."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        num_hands: int = 1,
        min_hand_detection_confidence: float = 0.5,
        min_hand_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        """Initialize MediaPipe HandLandmarker.

        Args:
            model_path: Path to hand_landmarker.task. Defaults to models/hand_landmarker.task.
            num_hands: Maximum number of hands to detect.
            min_hand_detection_confidence: Confidence threshold for hand detection.
            min_hand_presence_confidence: Confidence threshold for hand presence.
            min_tracking_confidence: Confidence threshold for tracking.
        """
        if model_path is None:
            # Look in standard repository locations
            repo_root = Path(__file__).resolve().parent.parent.parent
            default_path = repo_root / "models" / "hand_landmarker.task"
            if default_path.exists():
                model_path = str(default_path)
            else:
                raise FileNotFoundError(
                    f"MediaPipe task model not found at {default_path}. "
                    "Ensure models/hand_landmarker.task is present."
                )

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=num_hands,
            min_hand_detection_confidence=min_hand_detection_confidence,
            min_hand_presence_confidence=min_hand_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            running_mode=vision.RunningMode.VIDEO,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.frame_counter = 0
        self._last_ts_ms = -1

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        timestamp_sec: Optional[float] = None
    ) -> LandmarkFrame:
        """Process an OpenCV BGR image and extract 3D landmarks.

        Never raises on bad input: an empty / malformed frame, or a frame
        where no hand is found, returns ``LandmarkFrame(detected=False)``.

        Args:
            frame_bgr: OpenCV image array (H, W, 3) uint8 (grayscale and
                BGRA are converted automatically).
            timestamp_sec: Capture time in seconds (use the real clock, the
                physics engine differentiates with respect to this).
                Defaults to frame_idx / 30.

        Returns:
            LandmarkFrame dataclass with raw, world, normalized and metric
            coordinates.
        """
        if timestamp_sec is None:
            timestamp_sec = self.frame_counter / 30.0
        frame_idx = self.frame_counter
        self.frame_counter += 1

        def _missing() -> LandmarkFrame:
            return LandmarkFrame(
                frame_idx=frame_idx, timestamp_sec=timestamp_sec, detected=False
            )

        frame_bgr = self._prepare_frame(frame_bgr)
        if frame_bgr is None:
            return _missing()

        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # VIDEO mode needs strictly increasing integer millisecond timestamps.
        ts_ms = max(int(round(timestamp_sec * 1000.0)), self._last_ts_ms + 1)
        self._last_ts_ms = ts_ms
        result = self.detector.detect_for_video(mp_image, ts_ms)

        if not result.hand_landmarks:
            return _missing()

        # Take primary hand (index 0)
        raw_coords = np.array(
            [[lm.x, lm.y, lm.z] for lm in result.hand_landmarks[0]],
            dtype=np.float64,
        )
        if raw_coords.shape != (21, 3) or not np.all(np.isfinite(raw_coords)):
            return _missing()

        world_coords = None
        if result.hand_world_landmarks:
            world_coords = np.array(
                [[lm.x, lm.y, lm.z] for lm in result.hand_world_landmarks[0]],
                dtype=np.float64,
            )
            if world_coords.shape != (21, 3) or not np.all(np.isfinite(world_coords)):
                world_coords = None

        # Shape features: normalize in PIXEL space so x and y use the same
        # unit (normalized image coords are stretched by the aspect ratio).
        pixel_coords = self.to_pixel_coords(raw_coords, w, h)
        normalized_coords = self.normalize_coordinates(pixel_coords)
        metric_coords = self.to_metric_coords(pixel_coords, world_coords)

        handedness = "Unknown"
        confidence = 0.0
        if result.handedness and result.handedness[0]:
            category = result.handedness[0][0]
            handedness = category.category_name or "Unknown"
            confidence = float(category.score)

        return LandmarkFrame(
            frame_idx=frame_idx,
            timestamp_sec=timestamp_sec,
            detected=True,
            raw_landmarks=raw_coords.astype(np.float32),
            world_landmarks=world_coords.astype(np.float32) if world_coords is not None else None,
            normalized_landmarks=normalized_coords.astype(np.float32),
            metric_landmarks=metric_coords.astype(np.float32) if metric_coords is not None else None,
            handedness=handedness,
            confidence=confidence,
        )

    @staticmethod
    def _prepare_frame(frame) -> Optional[np.ndarray]:
        """Return a contiguous uint8 (H, W, 3) BGR frame, or None if unusable."""
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return None
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.ndim == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        elif frame.ndim != 3 or frame.shape[2] != 3:
            return None
        return np.ascontiguousarray(frame)

    @staticmethod
    def to_pixel_coords(raw_coords: np.ndarray, width: int, height: int) -> np.ndarray:
        """Convert MediaPipe image-normalized coords to pixel units.

        MediaPipe's z uses roughly the same scale as x, so z is multiplied by
        the width.
        """
        scale = np.array([width, height, width], dtype=np.float64)
        return np.asarray(raw_coords, dtype=np.float64) * scale

    @staticmethod
    def to_metric_coords(
        pixel_coords: np.ndarray, world_coords: Optional[np.ndarray]
    ) -> Optional[np.ndarray]:
        """Approximate metric (meter) coordinates that keep global hand motion.

        MediaPipe world landmarks are in meters but centered on the hand, so
        they lose whole-hand movement. Image landmarks keep movement but are
        in pixels. We convert pixels to meters with the ratio
        (wrist->middle-MCP length in meters) / (same length in pixels).
        This is approximate: it ignores camera perspective and is a bit off
        when the palm is strongly tilted toward the camera.

        Returns None if world landmarks are unavailable or degenerate.
        """
        if world_coords is None:
            return None
        world_palm = np.linalg.norm(world_coords[9] - world_coords[0])
        pixel_palm = np.linalg.norm(pixel_coords[9] - pixel_coords[0])
        if world_palm < 1e-6 or pixel_palm < 1e-6:
            return None
        return pixel_coords * (world_palm / pixel_palm)

    @staticmethod
    def normalize_coordinates(coords: np.ndarray) -> np.ndarray:
        """Normalize coordinates for scale and translation invariance.

        1. Translation invariance: Shifts origin (0, 0, 0) to Wrist (index 0).
        2. Scale invariance: Scales coordinates by the distance between
           Wrist (0) and Middle Finger MCP (9) (palm length).

        Args:
            coords: Array of shape (21, 3).

        Returns:
            Normalized coordinates array of shape (21, 3).
        """
        coords = np.asarray(coords, dtype=np.float64)
        if coords.shape != (21, 3):
            raise ValueError(f"Expected coords of shape (21, 3), got {coords.shape}")
        wrist = coords[0]
        centered = coords - wrist

        palm_length = np.linalg.norm(centered[9])  # Middle MCP is index 9
        if palm_length < 1e-6:
            scale_factor = 1.0
        else:
            scale_factor = palm_length

        return (centered / scale_factor).astype(np.float32)

    def close(self):
        """Release MediaPipe resources."""
        detector = getattr(self, "detector", None)
        if detector is not None:
            detector.close()
            self.detector = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
