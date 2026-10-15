"""Synthetic gesture dataset builder (metric landmarks, with participant ids).

Every gesture is a (T, 21, 3) trajectory in METRES (image-plane convention: +x right, +y down).
Classes share hand shape / path within pairs and differ in speed - the hard case for shape-only models:

    idle          still hand with tremor
    tap_gentle    open palm, downward strike,   peak ~0.35 m/s
    tap_forceful  open palm, downward strike,   peak ~1.2  m/s
    swipe_slow    pointing hand, sideways sweep, peak ~0.40 m/s
    swipe_rapid   pointing hand, sideways sweep, peak ~1.6  m/s

Each synthetic *participant* has their own speed bias and hand size so that participant-independent
evaluation (as required by the report) is meaningful.
"""

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple
import numpy as np

CLASS_NAMES: List[str] = ["idle", "tap_gentle", "tap_forceful", "swipe_slow", "swipe_rapid"]

# (x, y) of an upright open hand, wrist at origin, fingers toward -y (image coordinates), metres.
_OPEN = np.array([
    [0, 0],
    [-0.030, -0.015], [-0.058, -0.040], [-0.082, -0.062], [-0.100, -0.082],
    [-0.030, -0.090], [-0.033, -0.125], [-0.035, -0.148], [-0.036, -0.168],
    [-0.005, -0.095], [-0.005, -0.135], [-0.005, -0.160], [-0.005, -0.182],
    [0.020, -0.090], [0.022, -0.125], [0.023, -0.147], [0.024, -0.165],
    [0.040, -0.080], [0.044, -0.108], [0.046, -0.126], [0.047, -0.142],
], dtype=np.float64)

_SPECS: Dict[str, dict] = {
    "tap_gentle":   dict(axis=(0.0, 1.0), peak=0.35, amp=0.12, shape="open"),
    "tap_forceful": dict(axis=(0.0, 1.0), peak=1.20, amp=0.12, shape="open"),
    "swipe_slow":   dict(axis=(1.0, 0.0), peak=0.40, amp=0.15, shape="point"),
    "swipe_rapid":  dict(axis=(1.0, 0.0), peak=1.60, amp=0.25, shape="point"),
}


def hand_template(shape: str = "open") -> np.ndarray:
    """(21, 3) template. ``point`` folds middle/ring/pinky and the thumb toward the palm."""
    pts = np.zeros((21, 3))
    pts[:, :2] = _OPEN
    if shape == "point":
        for mcp in (9, 13, 17):                       # fold middle, ring, pinky
            base = pts[mcp, :2].copy()
            pts[mcp + 1, :2] = base + [0.0, -0.030]
            pts[mcp + 2, :2] = base + [0.0, -0.012]
            pts[mcp + 3, :2] = base + [0.0, 0.015]
            pts[mcp + 1:mcp + 4, 2] = [0.010, 0.025, 0.035]
        pts[2, :2] = [-0.040, -0.040]
        pts[3, :2] = [-0.030, -0.060]
        pts[4, :2] = [-0.015, -0.075]
    elif shape != "open":
        raise ValueError(f"unknown hand shape {shape!r}")
    return pts


def _min_jerk(tau: np.ndarray) -> np.ndarray:
    t = np.clip(tau, 0.0, 1.0)
    return t ** 3 * (10 - 15 * t + 6 * t ** 2)


@dataclass
class SyntheticDataset:
    trajectories: List[np.ndarray]
    labels: List[str]
    groups: List[int]                 # participant id per sample
    class_names: List[str]


class GestureDatasetBuilder:
    """Generates reproducible synthetic gesture trajectories."""

    @staticmethod
    def _one(label: str, rng: np.random.Generator, fps: float, frames: int, speed_bias: float, hand_scale: float) -> np.ndarray:
        T = int(frames)
        t = np.arange(T) / fps
        window_s = T / fps
        base_shape = ["open", "point"][int(rng.integers(0, 2))] if label == "idle" else _SPECS[label]["shape"]
        tmpl = hand_template(base_shape) * hand_scale
        offset = np.array([rng.uniform(-0.05, 0.05), rng.uniform(-0.03, 0.03), 0.0])

        disp = np.zeros((T, 3))
        if label != "idle":
            spec = _SPECS[label]
            peak = spec["peak"] * speed_bias * rng.uniform(0.85, 1.15)
            amp = spec["amp"] * rng.uniform(0.9, 1.1)
            dur = 1.875 * amp / peak                                   # min-jerk: v_peak = 1.875 A / T
            dur = min(dur, window_s - 0.06)
            start = rng.uniform(0.02, max(0.021, window_s - dur - 0.03))
            ang = np.deg2rad(rng.uniform(-8, 8))
            ax = np.array(spec["axis"]) @ np.array([[np.cos(ang), np.sin(ang)], [-np.sin(ang), np.cos(ang)]])
            prof = amp * _min_jerk((t - start) / dur)
            disp[:, 0], disp[:, 1] = prof * ax[0], prof * ax[1]
        else:
            drift = rng.normal(0, 0.004, size=2)                       # slow drift, << motion threshold
            disp[:, 0], disp[:, 1] = drift[0] * t, drift[1] * t

        traj = tmpl[None, :, :] + offset + disp[:, None, :]
        traj += rng.normal(0.0, 0.001, size=traj.shape)                # 1 mm landmark jitter
        return traj

    @staticmethod
    def generate_synthetic_dataset(
        samples_per_class: int = 30,
        fps: float = 30.0,
        frames_per_sample: int = 30,
        participants: int = 6,
        seed: int = 42,
        class_names: Sequence[str] = tuple(CLASS_NAMES),
    ) -> SyntheticDataset:
        if samples_per_class < 1 or participants < 1 or frames_per_sample < 10 or fps <= 0:
            raise ValueError("invalid dataset parameters")
        rng = np.random.default_rng(seed)
        bias = rng.uniform(0.9, 1.1, size=participants)                # per-participant speed style
        scale = rng.uniform(0.9, 1.1, size=participants)               # per-participant hand size
        trajs, labels, groups = [], [], []
        for i in range(samples_per_class):
            pid = i % participants
            for label in class_names:
                trajs.append(GestureDatasetBuilder._one(label, rng, fps, frames_per_sample, bias[pid], scale[pid]))
                labels.append(label)
                groups.append(pid)
        return SyntheticDataset(trajs, labels, groups, list(class_names))

    @staticmethod
    def generate_synthetic_gestures(
        samples_per_class: int = 30, fps: float = 30.0, frames_per_sample: int = 30, seed: int = 42,
    ) -> Tuple[List[np.ndarray], List[str], List[str]]:
        """Backwards-compatible API: ``(trajectories, labels, class_names)``."""
        ds = GestureDatasetBuilder.generate_synthetic_dataset(samples_per_class, fps, frames_per_sample, seed=seed)
        return ds.trajectories, ds.labels, ds.class_names
