"""Feature Aggregation and Shape-Physics Fusion.

Authored for the Data Engineering Workstream (Lead: Koushik).
Extracts geometric shape features and dynamic kinematic/dynamic features
from gesture trajectories to feed the Ablation Study classifiers.
"""

from typing import Dict, List, Tuple
import numpy as np
from src.physics.kinematics import KinematicsEngine, KinematicState
from src.physics.dynamics import DynamicsEngine, DynamicState


# Fingertip landmark indices: Thumb (4), Index (8), Middle (12), Ring (16), Pinky (20)
FINGERTIP_INDICES = [4, 8, 12, 16, 20]


class FeatureAggregator:
    """Extracts fused static shape and dynamic physical feature vectors."""

    def __init__(self, fps: float = 30.0):
        self.kinematics_engine = KinematicsEngine(default_fps=fps)
        self.dynamics_engine = DynamicsEngine()

    def extract_features(
        self,
        normalized_trajectory: np.ndarray,
        fps: float = 30.0
    ) -> Dict[str, np.ndarray]:
        """Extract shape-only, dynamics-only, and combined feature vectors.

        Args:
            normalized_trajectory: Array of shape (T, 21, 3) with wrist-normalized coordinates.
            fps: Video capture frame rate.

        Returns:
            Dictionary with keys:
                - 'shape': 1D feature vector of static geometry
                - 'dynamics': 1D feature vector of kinematic/energy metrics
                - 'combined': 1D concatenated vector of both
                - 'feature_names': List of feature names
        """
        traj = np.asarray(normalized_trajectory, dtype=np.float64)
        T, num_joints, _ = traj.shape

        if T < 3:
            raise ValueError(f"Trajectory length must be >= 3 frames (got {T}).")

        # --- 1. SHAPE FEATURES ---
        # Represent pose using mean normalized positions and key distances
        mean_pose = np.mean(traj, axis=0).flatten()  # 21 * 3 = 63 features
        shape_feature_list = list(mean_pose)

        # Fingertip-to-wrist Euclidean distances (at gesture midpoint)
        mid_frame = traj[T // 2]
        fingertip_dists = [np.linalg.norm(mid_frame[tip]) for tip in FINGERTIP_INDICES]
        shape_feature_list.extend(fingertip_dists)

        # Fingertip-to-thumb distances (finger closure/pinch indicator)
        thumb_tip = mid_frame[4]
        pinch_dists = [np.linalg.norm(mid_frame[tip] - thumb_tip) for tip in FINGERTIP_INDICES[1:]]
        shape_feature_list.extend(pinch_dists)

        shape_vector = np.array(shape_feature_list, dtype=np.float32)

        # --- 2. DYNAMICS & KINEMATICS FEATURES ---
        kin_state = self.kinematics_engine.analyze_trajectory(traj, fps=fps)
        dyn_state = self.dynamics_engine.compute_dynamics(kin_state)

        # Peak and mean velocity across whole hand
        peak_speed_overall = float(np.max(kin_state.speed))
        mean_speed_overall = float(np.mean(kin_state.speed))

        # Peak speeds per individual fingertip
        tip_peak_speeds = [float(np.max(kin_state.speed[:, tip])) for tip in FINGERTIP_INDICES]

        # Acceleration and Jerk peaks
        peak_accel = float(np.max(kin_state.accel_mag))
        mean_accel = float(np.mean(kin_state.accel_mag))
        peak_jerk = float(np.max(kin_state.jerk_mag))
        mean_jerk = float(np.mean(kin_state.jerk_mag))

        # Energy & Biomechanical metrics
        peak_ke = dyn_state.peak_kinetic_energy
        mean_ke = float(np.mean(dyn_state.kinetic_energy))
        total_ke = dyn_state.total_energy_expenditure
        peak_f = dyn_state.peak_force

        dynamics_features = [
            peak_speed_overall,
            mean_speed_overall,
            *tip_peak_speeds,
            peak_accel,
            mean_accel,
            peak_jerk,
            mean_jerk,
            peak_ke,
            mean_ke,
            total_ke,
            peak_f,
        ]
        dynamics_vector = np.array(dynamics_features, dtype=np.float32)

        # --- 3. COMBINED VECTOR ---
        combined_vector = np.concatenate([shape_vector, dynamics_vector])

        return {
            "shape": shape_vector,
            "dynamics": dynamics_vector,
            "combined": combined_vector,
        }
