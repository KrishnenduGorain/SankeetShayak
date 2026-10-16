"""Unit tests for Physics & Kinematics Engine (src/physics)."""

import numpy as np
import pytest
from src.physics.kinematics import KinematicsEngine, KinematicState
from src.physics.dynamics import DynamicsEngine, DynamicState


def test_constant_velocity_derivatives():
    """Verify that linear motion has constant velocity and near-zero acceleration/jerk."""
    fps = 30.0
    dt = 1.0 / fps
    T = 60
    t = np.linspace(0, (T - 1) * dt, T)

    # 21 joints, moving at constant vx = 2.0 m/s in X axis
    v_true = 2.0
    traj = np.zeros((T, 21, 3), dtype=np.float64)
    for i in range(21):
        traj[:, i, 0] = v_true * t

    engine = KinematicsEngine(default_fps=fps, savgol_window=7, savgol_polyorder=2)
    state = engine.analyze_trajectory(traj, fps=fps)

    # In interior frames (away from boundary effects), velocity should match v_true
    interior_v = state.velocities[5:-5, :, 0]
    np.testing.assert_allclose(interior_v, v_true, atol=0.05)

    # Acceleration and jerk should be practically zero
    interior_a = state.accelerations[5:-5, :, 0]
    np.testing.assert_allclose(interior_a, 0.0, atol=0.05)


def test_harmonic_motion_derivatives():
    """Verify differentiation against known sinusoidal ground truth: x = sin(t)."""
    fps = 60.0
    dt = 1.0 / fps
    T = 120
    t = np.linspace(0, (T - 1) * dt, T)

    omega = 2.0  # rad/s
    A = 1.5
    # x(t) = A * sin(omega * t)
    # v(t) = A * omega * cos(omega * t)
    # a(t) = -A * omega^2 * sin(omega * t)
    x = A * np.sin(omega * t)
    v_expected = A * omega * np.cos(omega * t)
    a_expected = -A * (omega ** 2) * np.sin(omega * t)

    traj = np.zeros((T, 21, 3), dtype=np.float64)
    traj[:, :, 0] = x[:, None]

    engine = KinematicsEngine(default_fps=fps, savgol_window=9, savgol_polyorder=3)
    state = engine.analyze_trajectory(traj, fps=fps)

    # Test interior slice
    slice_idx = slice(10, -10)
    np.testing.assert_allclose(state.velocities[slice_idx, 0, 0], v_expected[slice_idx], rtol=0.05)
    np.testing.assert_allclose(state.accelerations[slice_idx, 0, 0], a_expected[slice_idx], rtol=0.05)


def test_dynamics_kinetic_energy():
    """Verify kinetic energy is zero when stationary and positive when moving."""
    fps = 30.0
    T = 20
    traj_stationary = np.zeros((T, 21, 3), dtype=np.float64)

    kin_engine = KinematicsEngine(default_fps=fps)
    dyn_engine = DynamicsEngine()

    # Stationary test
    kin_stat = kin_engine.analyze_trajectory(traj_stationary, fps=fps)
    dyn_stat = dyn_engine.compute_dynamics(kin_stat)
    assert dyn_stat.peak_kinetic_energy == pytest.approx(0.0, abs=1e-7)

    # Moving test (rapid gesture)
    traj_moving = np.zeros((T, 21, 3), dtype=np.float64)
    traj_moving[:, :, 1] = np.linspace(0, 0.5, T)[:, None]  # 0.5m displacement
    kin_moving = kin_engine.analyze_trajectory(traj_moving, fps=fps)
    dyn_moving = dyn_engine.compute_dynamics(kin_moving)

    assert dyn_moving.peak_kinetic_energy > 0.0
    assert dyn_moving.total_energy_expenditure > 0.0
    assert dyn_moving.peak_force >= 0.0
