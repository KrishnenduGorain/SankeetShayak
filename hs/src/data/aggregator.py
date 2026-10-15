"""Feature aggregation: fuses hand-shape features with biomechanical dynamics features.

Input trajectories are (T, 21, 3) in METRES (see ``src.vision.metric``). Shape features are computed
after wrist-centring and palm-length normalisation, so they ignore where the hand is and how big it is;
dynamics features use the metric positions so they capture real motion.

The dynamics vector layout is FIXED (``GesturePredictor`` reads telemetry by index):
    0 peak_speed  1 mean_speed  2 speed_std  3 time_to_peak  4 peak_fingertip_speed  5 mean_fingertip_speed
    6 peak_deceleration  7 peak_accel  8 mean_accel  9 peak_jerk  10 rms_jerk  11 peak_ke  12 mean_ke
    13 total_energy  14 peak_force  15 peak_momentum  16 peak_torque  17 impulse  18 smoothness
    19 duration  20-22 direction (unit vector)  23 path_length
"""

from typing import Dict, List, Optional
import numpy as np

from src.physics.dynamics import FINGERTIP_INDICES, PALM_INDICES, DynamicsEngine
from src.physics.kinematics import KinematicsEngine

DYNAMICS_FEATURE_NAMES: List[str] = [
    "peak_speed", "mean_speed", "speed_std", "time_to_peak", "peak_fingertip_speed", "mean_fingertip_speed",
    "peak_deceleration", "peak_accel", "mean_accel", "peak_jerk", "rms_jerk", "peak_ke", "mean_ke",
    "total_energy", "peak_force", "peak_momentum", "peak_torque", "impulse", "smoothness", "duration",
    "dir_x", "dir_y", "dir_z", "path_length",
]

_FINGERS = [(1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)]
_ANGLE_TRIPLES: List[tuple] = []
for _i, (a, b, c, d) in enumerate(_FINGERS):
        _ANGLE_TRIPLES += [(0, a, b), (a, b, c), (b, c, d)]               # 3 bend angles per finger -> 15

SHAPE_FEATURE_NAMES: List[str] = (
    [f"lm{i}_{ax}" for i in range(1, 21) for ax in "xyz"]
    + [f"bend_{i}" for i in range(len(_ANGLE_TRIPLES))]
    + ["palm_nx", "palm_ny", "palm_nz", "palm_dx", "palm_dy", "palm_dz"]
)


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n < 1e-9, 1.0, n)


class FeatureAggregator:
    """Extracts shape, dynamics and combined feature vectors from a metric hand trajectory."""

    MIN_FRAMES = 5

    def __init__(self, fps: float = 30.0, hand_mass_kg: float = 0.40, savgol_window: int = 7, savgol_polyorder: int = 3):
        self.fps = fps
        self.kinematics = KinematicsEngine(default_fps=fps, savgol_window=savgol_window, savgol_polyorder=savgol_polyorder)
        self.dynamics = DynamicsEngine(hand_mass_kg=hand_mass_kg)

    # ------------------------------------------------------------------ shape
    @staticmethod
    def shape_features(traj: np.ndarray) -> np.ndarray:
        rel = traj - traj[:, :1, :]                                     # wrist origin
        palm = np.linalg.norm(rel[:, 9, :], axis=-1)                    # (T,)
        rel = rel / np.where(palm < 1e-9, 1.0, palm)[:, None, None]
        mean_lm = rel.mean(axis=0)[1:]                                  # (20, 3)

        bends = []
        for a, b, c in _ANGLE_TRIPLES:
            v1, v2 = _unit(rel[:, b, :] - rel[:, a, :]), _unit(rel[:, c, :] - rel[:, b, :])
            bends.append(np.arccos(np.clip(np.sum(v1 * v2, axis=-1), -1.0, 1.0)).mean())

        normal = _unit(np.cross(rel[:, 5, :], rel[:, 17, :])).mean(axis=0)
        direction = _unit(rel[:, 9, :]).mean(axis=0)
        return np.concatenate([mean_lm.ravel(), np.asarray(bends), _unit(normal), _unit(direction)])

    # ------------------------------------------------------------------ dynamics
    def dynamics_features(self, traj: np.ndarray, fps: float) -> np.ndarray:
        kin = self.kinematics.analyze_trajectory(traj, fps=fps)
        dyn = self.dynamics.compute_dynamics(kin)
        dt = kin.dt

        pv = kin.velocities[:, PALM_INDICES, :].mean(axis=1)            # (T, 3) palm-centre velocity
        speed = np.linalg.norm(pv, axis=-1)
        acc = np.linalg.norm(kin.accelerations[:, PALM_INDICES, :].mean(axis=1), axis=-1)
        jerk = np.linalg.norm(kin.jerks[:, PALM_INDICES, :].mean(axis=1), axis=-1)
        tips = kin.speed[:, FINGERTIP_INDICES].mean(axis=1)
        palm_pos = kin.positions[:, PALM_INDICES, :].mean(axis=1)

        peak = float(speed.max())
        i_peak = int(np.argmax(speed))
        dspeed = np.gradient(speed, dt)
        moving = speed > 0.25 * peak if peak > 0.10 else np.zeros_like(speed, dtype=bool)   # ignore tremor
        direction = _unit(pv[moving].mean(axis=0)) if moving.any() else np.zeros(3)
        duration = float(np.count_nonzero(speed > 0.10 * peak) * dt) if peak > 1e-6 else 0.0
        mom = np.linalg.norm(dyn.momentum, axis=-1)

        vec = np.array([
            peak, speed.mean(), speed.std(), i_peak * dt, float(tips.max()), float(tips.mean()),
            float(max(0.0, -dspeed.min())), float(acc.max()), float(acc.mean()), float(jerk.max()),
            float(np.sqrt(np.mean(jerk ** 2))), dyn.peak_kinetic_energy, float(dyn.kinetic_energy.mean()),
            dyn.total_energy_expenditure, dyn.peak_force, float(mom.max()), dyn.peak_torque, dyn.impulse,
            float(speed.mean() / peak) if peak > 1e-6 else 0.0, duration,
            direction[0], direction[1], direction[2],
            float(np.sum(np.linalg.norm(np.diff(palm_pos, axis=0), axis=-1))),
        ], dtype=np.float64)
        assert vec.shape[0] == len(DYNAMICS_FEATURE_NAMES)
        return vec

    # ------------------------------------------------------------------ public
    def extract_features(self, trajectory: np.ndarray, fps: Optional[float] = None) -> Dict[str, np.ndarray]:
        traj = np.asarray(trajectory, dtype=np.float64)
        if traj.ndim != 3 or traj.shape[1:] != (21, 3):
            raise ValueError(f"trajectory must have shape (T, 21, 3); got {traj.shape}")
        if traj.shape[0] < self.MIN_FRAMES:
            raise ValueError(f"need at least {self.MIN_FRAMES} frames, got {traj.shape[0]}")
        if not np.all(np.isfinite(traj)):
            raise ValueError("trajectory contains NaN or inf")
        fps = float(fps or self.fps)
        shape = np.nan_to_num(self.shape_features(traj))
        dynamics = np.nan_to_num(self.dynamics_features(traj, fps))
        return {"shape": shape, "dynamics": dynamics, "combined": np.concatenate([shape, dynamics])}
