"""Physics & Kinematics Engine.

Authored for the Physics Engine Workstream (Lead: Jai Krishna).
Computes numerical derivatives (velocity, acceleration, jerk) from 3D hand
trajectories using Savitzky–Golay polynomial smoothing filters.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from scipy.signal import savgol_filter


@dataclass
class KinematicState:
    """Kinematic metrics for a trajectory of hand landmarks."""
    positions: np.ndarray        # (T, 21, 3) Smoothed 3D positions
    velocities: np.ndarray       # (T, 21, 3) 3D velocity vectors
    accelerations: np.ndarray    # (T, 21, 3) 3D acceleration vectors
    jerks: np.ndarray            # (T, 21, 3) 3D jerk vectors
    speed: np.ndarray            # (T, 21) Scalar speed magnitudes
    accel_mag: np.ndarray        # (T, 21) Scalar acceleration magnitudes
    jerk_mag: np.ndarray         # (T, 21) Scalar jerk magnitudes
    dt: float                    # Sampling interval in seconds


class KinematicsEngine:
    """Calculates smoothed kinematic quantities for 3D landmark trajectories."""

    def __init__(
        self,
        default_fps: float = 30.0,
        savgol_window: int = 7,
        savgol_polyorder: int = 2,
    ):
        """Initialize Kinematics Engine.

        Args:
            default_fps: Default frame rate if explicit timestamps are not provided.
            savgol_window: Odd integer specifying the filter window length.
            savgol_polyorder: Degree of the polynomial used to fit the samples.
        """
        self.default_fps = default_fps
        self.default_dt = 1.0 / default_fps
        self.savgol_window = savgol_window
        self.savgol_polyorder = savgol_polyorder

    def analyze_trajectory(
        self,
        trajectory: np.ndarray,
        fps: Optional[float] = None,
        timestamps: Optional[np.ndarray] = None,
    ) -> KinematicState:
        """Compute smoothed kinematic derivatives for a trajectory of hand landmarks.

        Args:
            trajectory: Array of shape (T, 21, 3) or (T, 3) representing positions.
            fps: Frame rate for computing derivatives. Defaults to self.default_fps.
            timestamps: Optional array of timestamps (T,). If provided, calculates average dt.

        Returns:
            KinematicState containing positions, velocities, accelerations, and jerks.
        """
        traj = np.asarray(trajectory, dtype=np.float64)
        if traj.ndim == 2:
            # Expand (T, 3) to (T, 1, 3) for unified handling
            traj = np.expand_dims(traj, axis=1)

        T, num_joints, num_coords = traj.shape
        if T < 3:
            raise ValueError(f"Trajectory must have at least 3 frames for differentiation (got {T}).")

        # Determine sampling interval
        if timestamps is not None and len(timestamps) > 1:
            dt = float(np.mean(np.diff(timestamps)))
            if dt <= 0:
                dt = self.default_dt
        elif fps is not None and fps > 0:
            dt = 1.0 / fps
        else:
            dt = self.default_dt

        # Ensure Savitzky-Golay window does not exceed trajectory length and is odd
        window = min(self.savgol_window, T if T % 2 != 0 else T - 1)
        if window < 3:
            # Fall back to standard finite differences if sequence is too short for Savitzky-Golay
            smoothed = traj
            vel = np.gradient(smoothed, dt, axis=0)
            acc = np.gradient(vel, dt, axis=0)
            jerk = np.gradient(acc, dt, axis=0)
        else:
            polyorder = min(self.savgol_polyorder, window - 1)
            # 0th derivative: smoothed positions
            smoothed = savgol_filter(traj, window_length=window, polyorder=polyorder, deriv=0, axis=0)
            # 1st derivative: velocities
            vel = savgol_filter(traj, window_length=window, polyorder=polyorder, deriv=1, delta=dt, axis=0)
            # 2nd derivative: accelerations
            if polyorder >= 2:
                acc = savgol_filter(traj, window_length=window, polyorder=polyorder, deriv=2, delta=dt, axis=0)
            else:
                acc = np.gradient(vel, dt, axis=0)
            # 3rd derivative: jerk
            jerk = np.gradient(acc, dt, axis=0)

        # Compute Euclidean magnitudes across 3D coordinates
        speed = np.linalg.norm(vel, axis=-1)       # (T, num_joints)
        accel_mag = np.linalg.norm(acc, axis=-1)   # (T, num_joints)
        jerk_mag = np.linalg.norm(jerk, axis=-1)    # (T, num_joints)

        return KinematicState(
            positions=smoothed,
            velocities=vel,
            accelerations=acc,
            jerks=jerk,
            speed=speed,
            accel_mag=accel_mag,
            jerk_mag=jerk_mag,
            dt=dt,
        )

    # Aliases for interface compatibility
    compute_kinematics = analyze_trajectory
