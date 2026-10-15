"""Helpers that turn a stream of LandmarkFrame objects into clean arrays.

Downstream modules (physics, data, models) want a dense ``(T, 21, 3)`` array
and a timestamp vector. Real detections drop frames, so short gaps are
interpolated and long gaps are reported instead of silently filled.
"""

from typing import List, Optional, Tuple
import numpy as np

from src.vision.detector import LandmarkFrame


def frames_to_arrays(
    frames: List[LandmarkFrame], field: str = "metric_landmarks"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stack frames into arrays.

    Args:
        frames: Sequence of LandmarkFrame.
        field: Which landmark field to stack ("metric_landmarks",
            "normalized_landmarks", "raw_landmarks" or "world_landmarks").

    Returns:
        (trajectory (T, 21, 3) with NaN where missing, timestamps (T,),
        valid mask (T,) bool).
    """
    t = len(frames)
    traj = np.full((t, 21, 3), np.nan, dtype=np.float64)
    ts = np.zeros(t, dtype=np.float64)
    valid = np.zeros(t, dtype=bool)
    for i, f in enumerate(frames):
        ts[i] = f.timestamp_sec
        data = getattr(f, field, None)
        if f.detected and data is not None:
            traj[i] = data
            valid[i] = True
    return traj, ts, valid


def fill_gaps(
    traj: np.ndarray, valid: np.ndarray, max_gap: int = 3
) -> Tuple[np.ndarray, np.ndarray]:
    """Linearly interpolate short runs of missing frames.

    Gaps longer than ``max_gap`` frames, and gaps at the very start or end
    (nothing to interpolate between), stay NaN and stay marked invalid.

    Args:
        traj: (T, 21, 3) array, NaN where invalid.
        valid: (T,) bool mask of frames that have data.
        max_gap: Longest run of missing frames that will be filled.

    Returns:
        (filled trajectory copy, updated valid mask copy).
    """
    traj = np.array(traj, dtype=np.float64, copy=True)
    valid = np.array(valid, dtype=bool, copy=True)
    t = len(valid)
    i = 0
    while i < t:
        if valid[i]:
            i += 1
            continue
        j = i
        while j < t and not valid[j]:
            j += 1
        gap = j - i
        if i > 0 and j < t and gap <= max_gap:
            a, b = traj[i - 1], traj[j]
            for k in range(gap):
                w = (k + 1) / (gap + 1)
                traj[i + k] = (1.0 - w) * a + w * b
            valid[i:j] = True
        i = j
    return traj, valid


def longest_valid_run(valid: np.ndarray) -> Optional[Tuple[int, int]]:
    """Return (start, end_exclusive) of the longest run of True, or None."""
    best = None
    start = None
    for i, v in enumerate(list(valid) + [False]):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if best is None or (i - start) > (best[1] - best[0]):
                best = (start, i)
            start = None
    return best
