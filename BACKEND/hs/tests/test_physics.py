import numpy as np
import pytest

from src.physics.dynamics import DynamicsEngine
from src.physics.kinematics import KinematicsEngine


def _traj(f, T=30, fps=30.0):
    t = np.arange(T) / fps
    P = np.zeros((T, 21, 3))
    P[:, :, :] = np.arange(21)[None, :, None] * 0.01
    P[:, :, 0] += f(t)[:, None]
    return t, P


def test_polynomial_derivatives_are_exact():
    f = lambda t: 1 + 2 * t - 3 * t ** 2 + 4 * t ** 3
    t, P = _traj(f)
    k = KinematicsEngine(30.0).analyze_trajectory(P, fps=30.0)
    i = 15
    assert k.velocities[i, 0, 0] == pytest.approx(2 - 6 * t[i] + 12 * t[i] ** 2, rel=1e-6)
    assert k.accelerations[i, 0, 0] == pytest.approx(-6 + 24 * t[i], rel=1e-5)
    assert k.jerks[i, 0, 0] == pytest.approx(24, rel=1e-4)       # true 3rd derivative, not a gradient of noise


def test_irregular_timestamps_are_resampled():
    rng = np.random.default_rng(0)
    ts = np.cumsum(rng.uniform(0.02, 0.045, 40))
    P = np.zeros((40, 21, 3)); P[:, :, 0] = (0.5 * ts)[:, None]        # constant 0.5 m/s
    k = KinematicsEngine().analyze_trajectory(P, timestamps=ts)
    assert np.allclose(k.speed[5:-5].mean(), 0.5, rtol=0.02)


def test_validation():
    e = KinematicsEngine()
    with pytest.raises(ValueError):
        e.analyze_trajectory(np.zeros((2, 21, 3)))
    with pytest.raises(ValueError):
        e.analyze_trajectory(np.full((10, 21, 3), np.nan))
    with pytest.raises(ValueError):
        e.analyze_trajectory(np.zeros((10, 21, 2)))
    with pytest.raises(ValueError):
        DynamicsEngine(hand_mass_kg=0)


def test_short_trajectories_fall_back_gracefully():
    k = KinematicsEngine(savgol_window=11).analyze_trajectory(np.random.rand(4, 21, 3) * 0.01, fps=30)
    assert k.velocities.shape == (4, 21, 3) and np.isfinite(k.jerks).all()


def test_rigid_translation_energy_and_force():
    t, P = _traj(lambda t: 1.0 * t)                                     # 1 m/s, zero acceleration
    kin = KinematicsEngine().analyze_trajectory(P, fps=30.0)
    dyn = DynamicsEngine(0.4).compute_dynamics(kin)
    assert dyn.kinetic_energy[15] == pytest.approx(0.5 * 0.4 * 1.0, rel=1e-6)   # 1/2 M v^2 (all mass fractions normalised)
    assert dyn.peak_force < 1e-6 and dyn.kinetic_energy_rot[15] == pytest.approx(0, abs=1e-12)


def test_thin_rod_about_end_matches_textbook():
    # one segment rotating about its end: KE = 1/2 (mL^2/3) w^2 (translational CoM + rotation about CoM)
    w, L, m = 10.0, 0.1, 0.05
    T, fps = 21, 100.0
    t = np.arange(T) / fps
    P = np.zeros((T, 21, 3))
    P[:, 1, 0] = L * np.cos(w * t); P[:, 1, 1] = L * np.sin(w * t)
    for j in range(2, 21):
        P[:, j] = P[:, 0]
    eng = DynamicsEngine(m, segments=[(0, 1, 1.0)])
    kin = KinematicsEngine(fps, savgol_window=7, savgol_polyorder=3).analyze_trajectory(P, fps=fps)
    ke = eng.compute_dynamics(kin).kinetic_energy[10]
    assert ke == pytest.approx(0.5 * (m * L ** 2 / 3) * w ** 2, rel=2e-3)
