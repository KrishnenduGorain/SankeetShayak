"""Physics & Kinematics Engine.

Authored for the Physics Engine Workstream (Lead: Jai Krishna).
Computes numerical derivatives (velocity, acceleration, jerk) from 3D hand trajectories using
Savitzky-Golay polynomial smoothing filters.

Fixes over the first version:
* non-uniform frame timestamps (webcam jitter) are resampled to a uniform grid before filtering;
* jerk is the true 3rd Savitzky-Golay derivative (needs polyorder >= 3, now the default) instead of a
  finite difference of an already-noisy acceleration;
* inputs are validated (shape, finite values), and even/oversized windows are corrected.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
from scipy.signal import savgol_filter


@dataclass
class KinematicState:
    """Kinematic metrics for a trajectory of hand landmarks."""
    positions: np.ndarray        # (T, J, 3) Smoothed 3D positions (metres)
    velocities: np.ndarray       # (T, J, 3) m/s
    accelerations: np.ndarray    # (T, J, 3) m/s^2
    jerks: np.ndarray            # (T, J, 3) m/s^3
    speed: np.ndarray            # (T, J)
    accel_mag: np.ndarray        # (T, J)
    jerk_mag: np.ndarray         # (T, J)
    dt: float                    # Sampling interval in seconds


class KinematicsEngine:
    """Calculates smoothed kinematic quantities for 3D landmark trajectories."""

    def __init__(self, default_fps: float = 30.0, savgol_window: int = 7, savgol_polyorder: int = 3):
        if default_fps <= 0:
            raise ValueError("default_fps must be positive")
        self.default_fps = default_fps
        self.default_dt = 1.0 / default_fps
        self.savgol_window = savgol_window
        self.savgol_polyorder = savgol_polyorder

    @staticmethod
    def _resample_uniform(traj: np.ndarray, timestamps: np.ndarray):
        """Linearly resample (T, J, 3) data observed at ``timestamps`` onto a uniform grid. Returns (data, dt)."""
        t = np.asarray(timestamps, dtype=np.float64)
        order = np.argsort(t, kind="stable")
        t, traj = t[order], traj[order]
        keep = np.concatenate(([True], np.diff(t) > 1e-9))          # drop duplicate timestamps
        t, traj = t[keep], traj[keep]
        if len(t) < 3:
            raise ValueError("Need at least 3 distinct timestamps")
        dt = float(np.median(np.diff(t)))
        grid = t[0] + dt * np.arange(int(np.floor((t[-1] - t[0]) / dt + 1e-9)) + 1)
        flat = traj.reshape(len(t), -1)
        out = np.empty((len(grid), flat.shape[1]))
        for k in range(flat.shape[1]):
            out[:, k] = np.interp(grid, t, flat[:, k])
        return out.reshape((len(grid),) + traj.shape[1:]), dt

    def analyze_trajectory(
        self,
        trajectory: np.ndarray,
        fps: Optional[float] = None,
        timestamps: Optional[np.ndarray] = None,
    ) -> KinematicState:
        """Compute smoothed kinematic derivatives.

        Args:
            trajectory: (T, J, 3) or (T, 3) positions in metres.
            fps: frame rate (used when no timestamps are given).
            timestamps: optional (T,) seconds. If irregular, the data is resampled to a uniform grid
                (so the returned arrays may have a different length than the input).
        """
        traj = np.asarray(trajectory, dtype=np.float64)
        if traj.ndim == 2:
            traj = traj[:, np.newaxis, :]
        if traj.ndim != 3 or traj.shape[-1] != 3:
            raise ValueError(f"trajectory must be (T, J, 3) or (T, 3); got {traj.shape}")
        if not np.all(np.isfinite(traj)):
            raise ValueError("trajectory contains NaN or inf")
        T = traj.shape[0]
        if T < 3:
            raise ValueError(f"Trajectory must have at least 3 frames for differentiation (got {T}).")

        dt = self.default_dt if not (fps and fps > 0) else 1.0 / fps
        if timestamps is not None:
            ts = np.asarray(timestamps, dtype=np.float64)
            if ts.shape != (T,) or not np.all(np.isfinite(ts)):
                raise ValueError("timestamps must be a finite (T,) array")
            diffs = np.diff(ts)
            if len(diffs) and np.all(diffs > 0):
                mean = float(np.mean(diffs))
                if np.std(diffs) > 0.03 * mean:                    # jittery: resample
                    traj, dt = self._resample_uniform(traj, ts)
                else:
                    dt = mean
            elif len(diffs):
                traj, dt = self._resample_uniform(traj, ts)
            T = traj.shape[0]
            if T < 3:
                raise ValueError("Trajectory must have at least 3 frames after resampling")

        window = int(self.savgol_window)
        window = min(window, T if T % 2 == 1 else T - 1)
        if window % 2 == 0:
            window -= 1
        if window < 3:
            smoothed = traj
            vel = np.gradient(smoothed, dt, axis=0)
            acc = np.gradient(vel, dt, axis=0)
            jerk = np.gradient(acc, dt, axis=0)
        else:
            order = min(int(self.savgol_polyorder), window - 1)
            kw = dict(window_length=window, polyorder=order, delta=dt, axis=0, mode="interp")
            smoothed = savgol_filter(traj, deriv=0, **kw)
            vel = savgol_filter(traj, deriv=1, **kw)
            acc = savgol_filter(traj, deriv=2, **kw) if order >= 2 else np.gradient(vel, dt, axis=0)
            jerk = savgol_filter(traj, deriv=3, **kw) if order >= 3 else np.gradient(acc, dt, axis=0)

        return KinematicState(
            positions=smoothed, velocities=vel, accelerations=acc, jerks=jerk,
            speed=np.linalg.norm(vel, axis=-1), accel_mag=np.linalg.norm(acc, axis=-1),
            jerk_mag=np.linalg.norm(jerk, axis=-1), dt=float(dt),
        )
