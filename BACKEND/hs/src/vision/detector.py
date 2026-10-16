"""Hand Landmark Extraction (MediaPipe Tasks) with automatic model download and a test backend.

Authored for the Vision Pipeline (Lead: Debonnita); hardened for CI / cross-platform use.

Backends
--------
``tasks``      MediaPipe ``HandLandmarker`` (needs ``hand_landmarker.task``; downloaded on first use)
``synthetic``  deterministic scripted hand used by tests, CI and ``demo.py --selftest`` (no model, no camera)

The system rendering libraries MediaPipe / OpenCV need (libGL, libEGL, libGLESv2, GLib, X11) are
verified by :func:`src.system.render_env.ensure_render_env` *before* cv2 / mediapipe are imported.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Protocol, Tuple, Union

from src.system.render_env import ensure_render_env

ensure_render_env(strict=True)          # must run before the cv2 / mediapipe imports below

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from src.vision.metric import REF_PALM_LEN_M, MetricScaler  # noqa: E402

log = logging.getLogger(__name__)

HAND_LANDMARK_NAMES = [
    "WRIST",
    "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
]

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17), (0, 17),
]

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")
MODEL_FILENAME = "hand_landmarker.task"
_MIN_MODEL_BYTES = 1_000_000


class ModelUnavailableError(FileNotFoundError):
    """The MediaPipe hand_landmarker.task model is missing and could not be downloaded."""


# --------------------------------------------------------------------------- model asset handling
def default_model_path() -> Path:
    env = os.getenv("MEDIAPIPE_TASK_PATH")
    if env:
        p = Path(env)
        return p if p.is_absolute() else Path(__file__).resolve().parents[2] / p
    return Path(__file__).resolve().parents[2] / "models" / MODEL_FILENAME


def ensure_model(path: Optional[Union[str, Path]] = None, download: bool = True, retries: int = 3, timeout: float = 30.0) -> Path:
    """Return a valid model path, downloading it atomically if needed (set HANDSHAYAK_OFFLINE=1 to forbid)."""
    target = Path(path) if path else default_model_path()
    if target.exists() and target.stat().st_size >= _MIN_MODEL_BYTES:
        return target
    if not download or os.getenv("HANDSHAYAK_OFFLINE", "").lower() in {"1", "true", "yes"}:
        raise ModelUnavailableError(
            f"MediaPipe model not found at {target}. Run `python scripts/download_model.py` "
            f"or download {MODEL_URL} to that path.")
    target.parent.mkdir(parents=True, exist_ok=True)
    last: Optional[BaseException] = None
    for attempt in range(1, retries + 1):
        tmp_name = None
        try:
            with tempfile.NamedTemporaryFile(dir=target.parent, delete=False, suffix=".part") as tmp:
                tmp_name = tmp.name
                req = urllib.request.Request(MODEL_URL, headers={"User-Agent": "handshayak/1.0"})
                with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec - fixed https URL
                    shutil.copyfileobj(resp, tmp)
            if Path(tmp_name).stat().st_size < _MIN_MODEL_BYTES:
                raise OSError("downloaded file is too small to be a valid model")
            os.replace(tmp_name, target)          # atomic on Windows, macOS and Linux
            log.info("Downloaded hand model to %s", target)
            return target
        except Exception as exc:  # network errors, proxies, partial files
            last = exc
            if tmp_name and os.path.exists(tmp_name):
                try:
                    os.remove(tmp_name)
                except OSError:
                    pass
            time.sleep(min(2.0 * attempt, 6.0))
    raise ModelUnavailableError(f"Could not download hand model from {MODEL_URL}: {last}") from last


# --------------------------------------------------------------------------- data classes
@dataclass
class RawDetection:
    landmarks: np.ndarray                      # (21, 3) normalised image coords
    world: Optional[np.ndarray] = None         # (21, 3) metres, hand-centred
    handedness: str = "Unknown"
    score: float = 0.0


class DetectorBackend(Protocol):
    def detect(self, frame_rgb: np.ndarray) -> Optional[RawDetection]: ...
    def close(self) -> None: ...


@dataclass
class LandmarkFrame:
    """A single frame's detected hand landmarks."""
    frame_idx: int
    timestamp_sec: float
    detected: bool
    raw_landmarks: Optional[np.ndarray] = None          # (21, 3) normalised image coords
    world_landmarks: Optional[np.ndarray] = None        # (21, 3) metres, hand-centred (no global motion)
    normalized_landmarks: Optional[np.ndarray] = None   # (21, 3) wrist-origin, palm-length units (SHAPE ONLY)
    metric_landmarks: Optional[np.ndarray] = None       # (21, 3) metres, image-centre origin (USE FOR PHYSICS)
    meters_per_pixel: Optional[float] = None
    handedness: str = "Unknown"
    confidence: float = 0.0


# --------------------------------------------------------------------------- backends
class TasksBackend:
    """MediaPipe Tasks HandLandmarker (IMAGE mode: stateless, robust to irregular timestamps)."""

    def __init__(self, model_path: Optional[Union[str, Path]], num_hands: int, det: float, pres: float, track: float):
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        self._mp = mp
        path = ensure_model(model_path)
        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(path)),
            num_hands=num_hands,
            min_hand_detection_confidence=det,
            min_hand_presence_confidence=pres,
            min_tracking_confidence=track,
            running_mode=vision.RunningMode.IMAGE,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)

    def detect(self, frame_rgb: np.ndarray) -> Optional[RawDetection]:
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame_rgb))
        result = self._landmarker.detect(mp_image)
        if not result.hand_landmarks:
            return None
        lm = np.array([[p.x, p.y, p.z] for p in result.hand_landmarks[0]], dtype=np.float64)
        world = None
        if result.hand_world_landmarks:
            world = np.array([[p.x, p.y, p.z] for p in result.hand_world_landmarks[0]], dtype=np.float64)
        handed, score = "Unknown", 0.0
        if result.handedness and result.handedness[0]:
            cat = result.handedness[0][0]
            handed, score = cat.category_name, float(cat.score)
        return RawDetection(lm, world, handed, score)

    def close(self) -> None:
        try:
            self._landmarker.close()
        except Exception:  # pragma: no cover
            pass


class SyntheticBackend:
    """Replays a scripted (T, 21, 3) *metric* trajectory as if a camera were seeing it.

    Landmarks are projected to normalised image coordinates with a fixed palm size in pixels, so the
    real detector -> metric conversion -> physics path is exercised exactly as with a webcam.
    ``None`` is returned once the script ends (hand leaves the frame) unless ``loop`` is set.
    """

    def __init__(self, trajectory: np.ndarray, width: int = 640, height: int = 480, palm_px: float = 120.0, loop: bool = False):
        traj = np.asarray(trajectory, dtype=np.float64)
        if traj.ndim != 3 or traj.shape[1:] != (21, 3):
            raise ValueError(f"trajectory must be (T, 21, 3), got {traj.shape}")
        self.traj, self.width, self.height, self.loop = traj, width, height, loop
        self.m_per_px = REF_PALM_LEN_M / palm_px
        self.i = 0

    def detect(self, frame_rgb: np.ndarray) -> Optional[RawDetection]:
        if self.i >= len(self.traj):
            if not self.loop:
                return None
            self.i = 0
        P = self.traj[self.i]
        self.i += 1
        px = P / self.m_per_px
        lm = np.empty_like(px)
        lm[:, 0] = 0.5 + px[:, 0] / self.width
        lm[:, 1] = 0.5 + px[:, 1] / self.height
        lm[:, 2] = px[:, 2] / self.width
        return RawDetection(lm, P - P.mean(axis=0), "Right", 0.99)

    def close(self) -> None:
        pass


# --------------------------------------------------------------------------- detector
class HandLandmarkDetector:
    """Detects 21 3D hand landmarks in video frames (single hand)."""

    MAX_MISSED_BEFORE_RESET = 8

    def __init__(
        self,
        model_path: Optional[str] = None,
        num_hands: int = 1,
        min_hand_detection_confidence: float = 0.5,
        min_hand_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        backend: Union[str, DetectorBackend, None] = None,
    ):
        """
        Args:
            model_path: path to ``hand_landmarker.task`` (default ``models/hand_landmarker.task``; auto-downloaded).
            backend: ``None``/``"tasks"`` (MediaPipe), ``"synthetic"`` is created via :meth:`with_backend`,
                or any object implementing ``detect(frame_rgb)`` / ``close()``.
        """
        if backend is None:
            backend = os.getenv("HANDSHAYAK_BACKEND", "tasks")
        if isinstance(backend, str):
            if backend != "tasks":
                raise ValueError(f"Unknown backend {backend!r}; use 'tasks' or pass a backend object")
            backend = TasksBackend(model_path, num_hands, min_hand_detection_confidence,
                                   min_hand_presence_confidence, min_tracking_confidence)
        self.backend: DetectorBackend = backend
        self.scaler = MetricScaler()
        self.frame_counter = 0
        self._missed = 0

    @classmethod
    def with_backend(cls, backend: DetectorBackend) -> "HandLandmarkDetector":
        return cls(backend=backend)

    # ------------------------------------------------------------------ per frame
    def process_frame(self, frame_bgr: np.ndarray, timestamp_sec: Optional[float] = None) -> LandmarkFrame:
        if frame_bgr is None or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError("frame_bgr must be an (H, W, 3) BGR image")
        if timestamp_sec is None:
            timestamp_sec = self.frame_counter / 30.0
        h, w = frame_bgr.shape[:2]
        det = self.backend.detect(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

        idx = self.frame_counter
        self.frame_counter += 1
        if det is None:
            self._missed += 1
            if self._missed >= self.MAX_MISSED_BEFORE_RESET:
                self.scaler.reset()
            return LandmarkFrame(frame_idx=idx, timestamp_sec=timestamp_sec, detected=False)
        self._missed = 0

        raw = np.asarray(det.landmarks, dtype=np.float32)
        metric = self.scaler.to_metric(raw, w, h)
        if metric is None:                               # degenerate / tiny hand
            return LandmarkFrame(frame_idx=idx, timestamp_sec=timestamp_sec, detected=False)
        return LandmarkFrame(
            frame_idx=idx, timestamp_sec=timestamp_sec, detected=True,
            raw_landmarks=raw,
            world_landmarks=(det.world.astype(np.float32) if det.world is not None else raw),
            normalized_landmarks=self.normalize_coordinates(raw),
            metric_landmarks=metric.astype(np.float32),
            meters_per_pixel=self.scaler.meters_per_pixel,
            handedness=det.handedness, confidence=det.score,
        )

    @staticmethod
    def normalize_coordinates(coords: np.ndarray) -> np.ndarray:
        """Wrist-origin, palm-length-scaled coordinates (scale/translation invariant: use for SHAPE only)."""
        coords = np.asarray(coords, dtype=np.float64)
        centered = coords - coords[0]
        palm_length = np.linalg.norm(centered[9])
        return centered / (palm_length if palm_length >= 1e-6 else 1.0)

    # ------------------------------------------------------------------ batch extraction
    def extract_from_capture(
        self,
        cap: "cv2.VideoCapture",
        max_frames: Optional[int] = None,
        normalized: bool = True,
        interpolate_missing: bool = True,
        metric: bool = False,
    ) -> np.ndarray:
        """Extract landmarks across a ``cv2.VideoCapture``. Returns (num_frames, 21, 3).

        ``metric=True`` returns metres (use this for physics / training data); otherwise ``normalized``
        selects palm-normalised (True) or raw image coordinates (False).
        """
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if fps and fps > 1 else 30.0
        frames_data: List[Optional[np.ndarray]] = []
        self.scaler.reset()
        while cap.isOpened():
            if max_frames is not None and len(frames_data) >= max_frames:
                break
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            res = self.process_frame(frame, timestamp_sec=len(frames_data) / fps)
            if not res.detected:
                frames_data.append(None)
            elif metric:
                frames_data.append(res.metric_landmarks)
            else:
                frames_data.append(res.normalized_landmarks if normalized else res.raw_landmarks)

        out = np.zeros((len(frames_data), 21, 3), dtype=np.float32)
        last: Optional[np.ndarray] = None
        first_valid = next((f for f in frames_data if f is not None), None)
        if first_valid is None:
            return out
        last = first_valid
        for i, f in enumerate(frames_data):
            if f is not None:
                last = f
                out[i] = f
            elif interpolate_missing:
                out[i] = last
        return out

    def extract_from_video(self, video_path: Union[str, Path], max_frames: Optional[int] = None,
                           normalized: bool = True, interpolate_missing: bool = True, metric: bool = False) -> np.ndarray:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video file: {video_path}")
        try:
            return self.extract_from_capture(cap, max_frames, normalized, interpolate_missing, metric)
        finally:
            cap.release()

    def extract_from_webcam(self, duration_sec: float = 2.0, camera_index: int = 0, fps: float = 30.0,
                            normalized: bool = True, interpolate_missing: bool = True, metric: bool = False) -> np.ndarray:
        from src.vision.camera import CrossPlatformCamera      # per-OS backend selection (not AVFoundation everywhere)

        with CrossPlatformCamera(source=camera_index) as cam:
            return self.extract_from_capture(cam.cap, int(duration_sec * fps), normalized, interpolate_missing, metric)

    def close(self) -> None:
        if getattr(self, "backend", None) is not None:
            self.backend.close()

    def __enter__(self) -> "HandLandmarkDetector":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
