"""OpenCV rendering utilities that are safe on headless CI, Docker, macOS, Windows and Linux.

* :class:`SafeDisplay`  - ``imshow`` that degrades to a no-op instead of aborting the process.
* :class:`VideoSink`    - annotated-video writer with codec / container fallbacks.
* :func:`render_skeleton_canvas`, :func:`save_image` - offscreen rendering (no window, no GPU).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from src.system.render_env import can_display

log = logging.getLogger(__name__)

PathLike = Union[str, Path]


def _fourcc(code: str) -> int:
    # cv2.VideoWriter_fourcc exists in OpenCV 3/4/5 builds; VideoWriter.fourcc is the newer alias.
    fn = getattr(cv2, "VideoWriter_fourcc", None) or cv2.VideoWriter.fourcc
    return fn(*code)


# --------------------------------------------------------------------------- display
class SafeDisplay:
    """Window output that never crashes the process.

    ``cv2.imshow`` on a Linux box without an X server *aborts* (Qt platform plugin failure), so the
    display is only enabled after :func:`can_display` confirms a usable GUI. If ``cv2`` still raises, the
    display disables itself and the pipeline keeps running headless.
    """

    def __init__(self, title: str = "HandShayak", enabled: bool = True):
        self.title = title
        self.enabled = False
        self.reason = "display not requested"
        self._opened = False
        if enabled:
            ok, why = can_display()
            self.enabled, self.reason = ok, why
            if not ok:
                log.warning("Display disabled: %s", why)

    def show(self, frame: np.ndarray) -> bool:
        """Show a frame; returns False if no window is available."""
        if not self.enabled:
            return False
        try:
            cv2.imshow(self.title, frame)
            self._opened = True
            return True
        except cv2.error as exc:
            self.enabled, self.reason = False, f"cv2.imshow failed: {exc}"
            log.warning(self.reason)
            return False

    def quit_requested(self, delay_ms: int = 1) -> bool:
        """Pump the GUI event loop; True when the user pressed q or Esc."""
        if not (self.enabled and self._opened):
            return False
        try:
            key = cv2.waitKey(delay_ms) & 0xFF
        except cv2.error:
            return False
        return key in (ord("q"), 27)

    def close(self) -> None:
        if self._opened:
            try:
                cv2.destroyAllWindows()
                cv2.waitKey(1)  # lets macOS actually close the window
            except cv2.error:
                pass
            self._opened = False

    def __enter__(self) -> "SafeDisplay":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# --------------------------------------------------------------------------- video output
# (fourcc, extension) tried in order. mp4v works in the pip wheels on all three OSes; MJPG/AVI is the
# universal fallback because it needs no external codec at all.
_CODEC_CHAIN: Tuple[Tuple[str, str], ...] = (("mp4v", ".mp4"), ("MJPG", ".avi"), ("XVID", ".avi"))


class VideoSink:
    """Write BGR frames to a video file, falling back through codecs until one opens."""

    def __init__(self, path: PathLike, fps: float, size: Tuple[int, int], codecs: Sequence[Tuple[str, str]] = _CODEC_CHAIN):
        self.requested_path = Path(path)
        self.fps = float(fps) if fps and fps > 1 else 30.0
        self.size = (int(size[0]), int(size[1]))
        if self.size[0] <= 0 or self.size[1] <= 0:
            raise ValueError(f"Invalid frame size {size}")
        self.path: Optional[Path] = None
        self._writer: Optional[cv2.VideoWriter] = None
        self.frames_written = 0
        self.requested_path.parent.mkdir(parents=True, exist_ok=True)
        # mp4v / MJPG require even dimensions in some builds; we resize frames on write if needed.
        for code, ext in codecs:
            candidate = self.requested_path.with_suffix(ext) if self.requested_path.suffix.lower() != ext else self.requested_path
            writer = cv2.VideoWriter(str(candidate), _fourcc(code), self.fps, self.size)
            if writer.isOpened():
                self._writer, self.path, self.codec = writer, candidate, code
                break
            writer.release()
        if self._writer is None:
            raise RuntimeError(f"No usable video codec for {self.requested_path} (tried {[c for c, _ in codecs]})")

    def write(self, frame_bgr: np.ndarray) -> None:
        if self._writer is None:
            raise RuntimeError("VideoSink is closed")
        if frame_bgr.dtype != np.uint8:
            frame_bgr = np.clip(frame_bgr, 0, 255).astype(np.uint8)
        if frame_bgr.shape[1] != self.size[0] or frame_bgr.shape[0] != self.size[1]:
            frame_bgr = cv2.resize(frame_bgr, self.size, interpolation=cv2.INTER_AREA)
        self._writer.write(frame_bgr)
        self.frames_written += 1

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    def __enter__(self) -> "VideoSink":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# --------------------------------------------------------------------------- offscreen rendering
def blank_canvas(width: int = 640, height: int = 480, color: Tuple[int, int, int] = (20, 20, 20)) -> np.ndarray:
    canvas = np.empty((height, width, 3), dtype=np.uint8)
    canvas[:] = color
    return canvas


def render_skeleton_canvas(landmarks_norm: np.ndarray, width: int = 640, height: int = 480) -> np.ndarray:
    """Render a hand skeleton from normalised [0,1] image landmarks onto a blank canvas (no window needed)."""
    from src.vision.visualizer import HandVisualizer  # local import: visualizer imports detector constants

    return HandVisualizer().draw_skeleton(blank_canvas(width, height), np.asarray(landmarks_norm, dtype=np.float64))


def save_image(path: PathLike, image_bgr: np.ndarray) -> Path:
    """cv2.imwrite with directory creation and a hard failure instead of a silent ``False``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix or ".png", image_bgr)  # imencode+write handles non-ASCII paths on Windows
    if not ok:
        raise RuntimeError(f"Could not encode image for {path}")
    buf.tofile(str(path))
    return path


def read_all_frames(path: PathLike, limit: Optional[int] = None):
    """Decode a video file into a list of BGR frames (test / tooling helper)."""
    cap = cv2.VideoCapture(str(path))
    frames = []
    try:
        while cap.isOpened() and (limit is None or len(frames) < limit):
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            frames.append(frame)
    finally:
        cap.release()
    return frames
