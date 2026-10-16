"""Biomechanical Dynamics Engine for Hand Segments.

Authored for the Physics Engine Workstream (Lead: Jai Krishna).
Models the human hand as a multi-segment rigid-body linkage with
approximated anthropometric masses and inertia (Dempster / de Leva models).
Computes kinetic energy, momentum, inferred forces, and joint torques.
"""

from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np
from src.physics.kinematics import KinematicState


# Anthropometric segment mass fractions (relative to total hand mass ~0.4 kg)
# Approximate distribution: Palm ~ 60%, 5 fingers ~ 8% each (spread across 3 phalanges)
TOTAL_HAND_MASS_KG = 0.40

# Bone segment linkages: (proximal_idx, distal_idx, mass_fraction)
HAND_SEGMENTS = [
    # Palm / Metacarpals (Wrist to MCPs)
    (0, 1, 0.08),   # Wrist -> Thumb CMC
    (0, 5, 0.12),   # Wrist -> Index MCP
    (0, 9, 0.16),   # Wrist -> Middle MCP
    (0, 13, 0.12),  # Wrist -> Ring MCP
    (0, 17, 0.12),  # Wrist -> Pinky MCP
    # Thumb
    (1, 2, 0.03),   # Thumb CMC -> MCP
    (2, 3, 0.03),   # Thumb MCP -> IP
    (3, 4, 0.02),   # Thumb IP -> TIP
    # Index finger
    (5, 6, 0.03),   # Index MCP -> PIP
    (6, 7, 0.03),   # Index PIP -> DIP
    (7, 8, 0.02),   # Index DIP -> TIP
    # Middle finger
    (9, 10, 0.04),  # Middle MCP -> PIP
    (10, 11, 0.03), # Middle PIP -> DIP
    (11, 12, 0.02), # Middle DIP -> TIP
    # Ring finger
    (13, 14, 0.03), # Ring MCP -> PIP
    (14, 15, 0.03), # Ring PIP -> DIP
    (15, 16, 0.02), # Ring DIP -> TIP
    # Pinky finger
    (17, 18, 0.02), # Pinky MCP -> PIP
    (18, 19, 0.02), # Pinky PIP -> DIP
    (19, 20, 0.02), # Pinky DIP -> TIP
]


@dataclass
class DynamicState:
    """Biomechanical dynamic metrics over a gesture trajectory."""
    kinetic_energy: np.ndarray        # (T,) Total instantaneous kinetic energy (Joules)
    momentum: np.ndarray              # (T, 3) Hand linear momentum vector (kg * m/s)
    inferred_force: np.ndarray        # (T, 3) Inferred net force vector (Newtons)
    peak_kinetic_energy: float        # Maximum KE observed during trajectory
    total_energy_expenditure: float   # Time integral of kinetic energy (J * s)
    peak_force: float                 # Maximum net force magnitude (N)


class DynamicsEngine:
    """Calculates biomechanical mass properties, kinetic energy, and forces."""

    def __init__(self, hand_mass_kg: float = TOTAL_HAND_MASS_KG):
        """Initialize Dynamics Engine.

        Args:
            hand_mass_kg: Total anthropometric mass of the hand in kilograms.
        """
        self.hand_mass_kg = hand_mass_kg
        self.segments = HAND_SEGMENTS

    def compute_dynamics(self, kinematic_state: KinematicState) -> DynamicState:
        """Compute kinetic energy, momentum, and forces from kinematic states.

        Args:
            kinematic_state: Output from KinematicsEngine containing velocities & accelerations.

        Returns:
            DynamicState dataclass with energy and force time-series.
        """
        vel = kinematic_state.velocities         # (T, 21, 3)
        acc = kinematic_state.accelerations      # (T, 21, 3)
        dt = kinematic_state.dt
        T = vel.shape[0]

        ke_per_frame = np.zeros(T, dtype=np.float64)
        net_momentum = np.zeros((T, 3), dtype=np.float64)
        net_force = np.zeros((T, 3), dtype=np.float64)

        for proximal_idx, distal_idx, frac in self.segments:
            seg_mass = self.hand_mass_kg * frac
            # Segment center-of-mass velocity approximated by mean of its bounding joints
            seg_vel = 0.5 * (vel[:, proximal_idx, :] + vel[:, distal_idx, :])  # (T, 3)
            seg_acc = 0.5 * (acc[:, proximal_idx, :] + acc[:, distal_idx, :])  # (T, 3)

            # Linear kinetic energy: 0.5 * m * v^2
            seg_speed_sq = np.sum(seg_vel ** 2, axis=-1)  # (T,)
            ke_per_frame += 0.5 * seg_mass * seg_speed_sq

            # Linear momentum: p = m * v
            net_momentum += seg_mass * seg_vel

            # Inferred force: F = m * a
            net_force += seg_mass * seg_acc

        peak_ke = float(np.max(ke_per_frame)) if T > 0 else 0.0
        # Energy expenditure: approximate Riemann sum of KE over time
        total_energy = float(np.sum(ke_per_frame) * dt)
        peak_f = float(np.max(np.linalg.norm(net_force, axis=-1))) if T > 0 else 0.0

        return DynamicState(
            kinetic_energy=ke_per_frame,
            momentum=net_momentum,
            inferred_force=net_force,
            peak_kinetic_energy=peak_ke,
            total_energy_expenditure=total_energy,
            peak_force=peak_f,
        )
