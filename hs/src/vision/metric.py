"""Pixel -> metre conversion for MediaPipe landmarks (no cv2 dependency).

The original pipeline fed *wrist-centred, palm-scaled* landmarks to the physics engine. Those have zero
global translation (the wrist is always the origin) and are measured in palm lengths, not metres, so
tap / strike speeds were always ~0 and every "m/s" figure was wrong. Physics needs positions that
move with the hand and are expressed in metres.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

REF_PALM_LEN_M = 0.095   # wrist -> middle-finger MCP, adult average (anthropometric tables)


class MetricScaler:
    """Converts normalised image landmarks (x,y in [0,1], z in x-units) into metres.

    Scale (metres per pixel) comes from the hand's own 3D palm length, which is invariant to hand
    rotation, and is low-pass filtered so frame-to-frame jitter does not look like motion.
    Origin is the image centre; +x right, +y down, +z away from the camera (MediaPipe convention).
    """

    def __init__(self, ref_len_m: float = REF_PALM_LEN_M, alpha: float = 0.08, min_palm_px: float = 8.0):
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        self.ref_len_m = ref_len_m
        self.alpha = alpha
        self.min_palm_px = min_palm_px
        self.scale: Optional[float] = None       # metres per pixel (smoothed)

    def reset(self) -> None:
        self.scale = None

    @property
    def meters_per_pixel(self) -> Optional[float]:
        return self.scale

    def to_metric(self, landmarks: np.ndarray, width: int, height: int) -> Optional[np.ndarray]:
        """Return (21, 3) metric landmarks, or None if the hand is too small / degenerate to scale."""
        lm = np.asarray(landmarks, dtype=np.float64)
        if lm.shape != (21, 3) or not np.all(np.isfinite(lm)) or width <= 0 or height <= 0:
            return None
        px = lm * np.array([width, height, width], dtype=np.float64)
        palm_px = float(np.linalg.norm(px[0] - px[9]))
        if palm_px < self.min_palm_px:
            return None
        raw = self.ref_len_m / palm_px
        self.scale = raw if self.scale is None else self.scale + self.alpha * (raw - self.scale)
        out = px.copy()
        out[:, 0] -= width / 2.0
        out[:, 1] -= height / 2.0
        return out * self.scale
