"""Cross-Platform Camera & Video Capture Utility.

Handles video streaming and device acquisition seamlessly across
macOS (AVFoundation), Windows (DirectShow/MSMF), and Linux (V4L2).
"""

from pathlib import Path
import sys
from typing import Generator, List, Optional, Tuple, Union
import cv2
import numpy as np


def get_platform_camera_backends() -> List[int]:
    """Return prioritized OpenCV capture backends based on operating system."""
    platform = sys.platform.lower()

    if platform == "darwin":  # macOS
        return [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
    elif platform.startswith("win"):  # Windows
        # DirectShow is fastest & lowest-latency on Windows; fallback to MSMF & ANY
        return [getattr(cv2, "CAP_DSHOW", cv2.CAP_ANY), getattr(cv2, "CAP_MSMF", cv2.CAP_ANY), cv2.CAP_ANY]
    elif platform.startswith("linux"):  # Linux
        return [getattr(cv2, "CAP_V4L2", cv2.CAP_ANY), cv2.CAP_ANY]
    else:
        return [cv2.CAP_ANY]


def get_camera_permission_guidance() -> str:
    """Return OS-specific guidance for resolving camera permission issues."""
    platform = sys.platform.lower()

    if platform == "darwin":
        return (
            "macOS Camera Permission:\n"
            "1. Open System Settings > Privacy & Security > Camera.\n"
            "2. Ensure your Terminal/IDE (Terminal, iTerm2, VS Code, Cursor) is toggled ON.\n"
            "3. Restart the terminal after changing permissions."
        )
    elif platform.startswith("win"):
        return (
            "Windows Camera Permission:\n"
            "1. Open Settings > Privacy & security > Camera.\n"
            "2. Ensure 'Camera access' and 'Let desktop apps access your camera' are turned ON.\n"
            "3. Check that no other application (Zoom, Teams, Skype) is exclusively holding the camera."
        )
    elif platform.startswith("linux"):
        return (
            "Linux Camera Access:\n"
            "1. Check if device exists: ls -l /dev/video*\n"
            "2. Add your user to video group: sudo usermod -a -G video $USER\n"
            "3. Relog or run: newgrp video"
        )
    return "Ensure camera device is connected and not in use by another program."


def probe_available_cameras(max_indices: int = 3) -> List[int]:
    """Test available camera indices on the system across supported backends.

    Args:
        max_indices: Maximum number of device indices to probe (default: 3).

    Returns:
        List of working camera indices.
    """
    working_indices: List[int] = []
    backends = get_platform_camera_backends()

    for idx in range(max_indices):
        opened = False
        for backend in backends:
            cap = cv2.VideoCapture(idx, backend)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    working_indices.append(idx)
                    opened = True
                cap.release()
            if opened:
                break

    return working_indices


class CrossPlatformCamera:
    """Cross-platform camera and video file capture context manager.

    Works identically on macOS, Windows, and Linux.
    """

    def __init__(
        self,
        source: Union[int, str, Path] = 0,
        target_width: Optional[int] = None,
        target_height: Optional[int] = None,
        target_fps: Optional[float] = None,
    ):
        """Initialize camera or video stream.

        Args:
            source: Integer camera index (0, 1, 2) OR path to a video file.
            target_width: Desired capture width (hardware hint for webcams).
            target_height: Desired capture height (hardware hint for webcams).
            target_fps: Desired capture FPS.
        """
        self.source = source
        self.is_video_file = isinstance(source, (str, Path))
        self.target_width = target_width
        self.target_height = target_height
        self.target_fps = target_fps
        self.cap: Optional[cv2.VideoCapture] = None

    def open(self) -> "CrossPlatformCamera":
        """Open the camera device or video file using optimal backend."""
        if self.is_video_file:
            path_str = str(self.source)
            if not Path(path_str).exists():
                raise FileNotFoundError(f"Video file not found at: {path_str}")
            self.cap = cv2.VideoCapture(path_str)
            if not self.cap.isOpened():
                raise RuntimeError(f"OpenCV failed to open video file: {path_str}")
        else:
            cam_idx = int(self.source)
            backends = get_platform_camera_backends()
            opened = False

            # Try candidate backends
            for backend in backends:
                cap = cv2.VideoCapture(cam_idx, backend)
                if cap.isOpened():
                    # Test frame read to guarantee authorization & hardware availability
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        self.cap = cap
                        opened = True
                        break
                    cap.release()

            if not opened:
                # Fallback to default backend
                cap = cv2.VideoCapture(cam_idx)
                if cap.isOpened():
                    self.cap = cap
                    opened = True
                else:
                    cap.release()

            if not opened or self.cap is None:
                avail = probe_available_cameras()
                guidance = get_camera_permission_guidance()
                raise RuntimeError(
                    f"Could not open camera device index {cam_idx}.\n"
                    f"Available detected camera indices: {avail}\n"
                    f"Troubleshooting:\n{guidance}"
                )

            # Apply hardware camera hints
            if self.target_width is not None:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.target_width))
            if self.target_height is not None:
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.target_height))
            if self.target_fps is not None:
                self.cap.set(cv2.CAP_PROP_FPS, float(self.target_fps))

        return self

    def is_opened(self) -> bool:
        """Check if capture is open and ready."""
        return self.cap is not None and self.cap.isOpened()

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a single frame from the stream.

        Returns:
            Tuple of (success_flag, frame_bgr_array).
        """
        if not self.is_opened():
            return False, None
        return self.cap.read()

    @property
    def fps(self) -> float:
        """Frame rate property alias."""
        return self.get_fps()

    def get_fps(self) -> float:
        """Get capture frame rate."""
        if not self.is_opened():
            return 30.0
        fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        return fps if fps > 1.0 else 30.0

    def get_resolution(self) -> Tuple[int, int]:
        """Get current width and height of video frames."""
        if not self.is_opened():
            return (0, 0)
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (w, h)

    def release(self):
        """Release underlying OpenCV capture handle."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def frames(self, max_frames: Optional[int] = None) -> Generator[Tuple[int, np.ndarray], None, None]:
        """Yield frames sequentially with frame index.

        Args:
            max_frames: Optional limit on number of frames.

        Yields:
            (frame_idx, frame_bgr).
        """
        idx = 0
        while self.is_opened():
            if max_frames is not None and idx >= max_frames:
                break
            ret, frame = self.read()
            if not ret or frame is None:
                break
            yield idx, frame
            idx += 1

    def stream(self, max_frames: Optional[int] = None) -> Generator[np.ndarray, None, None]:
        """Yield raw frames sequentially without index for direct iteration."""
        for _, frame in self.frames(max_frames=max_frames):
            yield frame

    def __enter__(self) -> "CrossPlatformCamera":
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
