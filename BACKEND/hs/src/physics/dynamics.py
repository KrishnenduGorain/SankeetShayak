"""Biomechanical Dynamics Engine for Hand Segments.

Authored for the Physics Engine Workstream (Lead: Jai Krishna).
Models the hand as a multi-segment rigid-body linkage with anthropometric masses (Dempster / de Leva).
Computes translational + rotational kinetic energy, momentum, inferred net force, simplified joint
torque and impulse (equations 6-15 of the project report).

Mass fractions are re-normalised to sum to 1 so total segment mass equals ``hand_mass_kg``.
Force / torque are *inferred* from video motion and assumed masses, not measured.
"""

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple
import numpy as np
from src.physics.kinematics import KinematicState

TOTAL_HAND_MASS_KG = 0.40
PALM_INDICES = (0, 5, 9, 13, 17)
FINGERTIP_INDICES = (4, 8, 12, 16, 20)

# (proximal_idx, distal_idx, mass_fraction)
HAND_SEGMENTS: Sequence[Tuple[int, int, float]] = [
    (0, 1, 0.08), (0, 5, 0.12), (0, 9, 0.16), (0, 13, 0.12), (0, 17, 0.12),
    (1, 2, 0.03), (2, 3, 0.03), (3, 4, 0.02),
    (5, 6, 0.03), (6, 7, 0.03), (7, 8, 0.02),
    (9, 10, 0.04), (10, 11, 0.03), (11, 12, 0.02),
    (13, 14, 0.03), (14, 15, 0.03), (15, 16, 0.02),
    (17, 18, 0.02), (18, 19, 0.02), (19, 20, 0.02),
]


@dataclass
class DynamicState:
    """Biomechanical dynamic metrics over a gesture trajectory."""
    kinetic_energy: np.ndarray        # (T,) total KE = translational + rotational (J)
    momentum: np.ndarray              # (T, 3) kg*m/s
    inferred_force: np.ndarray        # (T, 3) N
    peak_kinetic_energy: float
    total_energy_expenditure: float   # integral of KE dt (J*s)
    peak_force: float                 # N
    kinetic_energy_rot: np.ndarray = field(default_factory=lambda: np.zeros(0))   # (T,) rotational part (J)
    torque: np.ndarray = field(default_factory=lambda: np.zeros(0))               # (T,) max segment torque (N*m)
    peak_torque: float = 0.0
    impulse: float = 0.0              # |integral F dt| (N*s)


class DynamicsEngine:
    """Calculates biomechanical mass properties, kinetic energy, and forces."""

    def __init__(self, hand_mass_kg: float = TOTAL_HAND_MASS_KG, segments: Optional[Sequence[Tuple[int, int, float]]] = None):
        if hand_mass_kg <= 0:
            raise ValueError("hand_mass_kg must be positive")
        self.hand_mass_kg = float(hand_mass_kg)
        self.segments = list(segments) if segments is not None else list(HAND_SEGMENTS)

    def compute_dynamics(self, kinematic_state: KinematicState) -> DynamicState:
        pos, vel, acc = kinematic_state.positions, kinematic_state.velocities, kinematic_state.accelerations
        dt = kinematic_state.dt
        T = vel.shape[0]
        if vel.shape[1] < 21:
            raise ValueError("DynamicsEngine needs 21 hand landmarks per frame")
        frac_sum = sum(f for _, _, f in self.segments)

        ke_trans = np.zeros(T)
        ke_rot = np.zeros(T)
        momentum = np.zeros((T, 3))
        force = np.zeros((T, 3))
        torque = np.zeros(T)

        for p, d, frac in self.segments:
            m = self.hand_mass_kg * frac / frac_sum
            r = pos[:, d, :] - pos[:, p, :]                       # segment vector
            L2 = np.sum(r * r, axis=-1)                           # (T,)
            v_c = 0.5 * (vel[:, p, :] + vel[:, d, :])             # centre-of-mass velocity
            a_c = 0.5 * (acc[:, p, :] + acc[:, d, :])
            ke_trans += 0.5 * m * np.sum(v_c ** 2, axis=-1)       # eq. (8)
            momentum += m * v_c                                   # eq. (11)
            force += m * a_c                                      # eq. (13)

            valid = L2 > 1e-10
            safe_L2 = np.where(valid, L2, 1.0)
            omega = np.where(valid, np.linalg.norm(np.cross(r, vel[:, d, :] - vel[:, p, :]), axis=-1) / safe_L2, 0.0)
            alpha = np.where(valid, np.linalg.norm(np.cross(r, acc[:, d, :] - acc[:, p, :]), axis=-1) / safe_L2, 0.0)
            I_c = m * L2 / 12.0                                   # eq. (6)
            ke_rot += 0.5 * I_c * omega ** 2                      # eq. (9)
            torque = np.maximum(torque, I_c * alpha)              # eq. (14)

        ke = ke_trans + ke_rot
        return DynamicState(
            kinetic_energy=ke, momentum=momentum, inferred_force=force,
            peak_kinetic_energy=float(ke.max()) if T else 0.0,
            total_energy_expenditure=float(np.sum(ke) * dt),
            peak_force=float(np.linalg.norm(force, axis=-1).max()) if T else 0.0,
            kinetic_energy_rot=ke_rot, torque=torque,
            peak_torque=float(torque.max()) if T else 0.0,
            impulse=float(np.linalg.norm(force.sum(axis=0) * dt)),   # eq. (15)
        )
