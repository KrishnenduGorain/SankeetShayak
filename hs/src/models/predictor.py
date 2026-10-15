"""Sign Gesture Inference Engine.

Authored for the ML & Backend Workstream (Lead: Harsimran).
Fast inference on sliding-window hand trajectories (metres, shape (T, 21, 3)).
"""

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Dict, Optional
import joblib
import numpy as np

from src.data.aggregator import FeatureAggregator

log = logging.getLogger(__name__)

WINDOW_FRAMES = 30


@dataclass
class GesturePrediction:
    """Prediction result and accompanying physical telemetry."""
    label: str
    confidence: float
    probabilities: Dict[str, float]
    telemetry: Dict[str, float]


class GesturePredictor:
    """Predicts sign labels from sliding-window coordinate trajectories."""

    def __init__(self, model_path: Optional[str] = None, fps: float = 30.0, window_frames: int = WINDOW_FRAMES):
        self.fps = fps
        self.window_frames = window_frames
        self.aggregator = FeatureAggregator(fps=fps)

        if model_path is None:
            from src.config import settings
            self.model_path = settings.model_path
        else:
            self.model_path = Path(model_path)
        self.model = self._load_or_train()

    def _load_or_train(self):
        from src.models.train import build_and_train_model
        if self.model_path.exists():
            try:
                return joblib.load(self.model_path)
            except Exception as exc:       # corrupt file / scikit-learn version change
                log.warning("Could not load %s (%s); retraining.", self.model_path, exc)
        return build_and_train_model(output_model_path=str(self.model_path), verbose=False)

    def _fit_window(self, traj: np.ndarray) -> np.ndarray:
        """Match the training window length: keep the latest frames, or pad the start with the first frame."""
        T = traj.shape[0]
        if T > self.window_frames:
            return traj[-self.window_frames:]
        if T < self.window_frames:
            pad = np.repeat(traj[:1], self.window_frames - T, axis=0)
            return np.concatenate([pad, traj], axis=0)
        return traj

    def predict_trajectory(self, trajectory: np.ndarray, fps: Optional[float] = None) -> GesturePrediction:
        """Run inference on a gesture trajectory of shape (T, 21, 3) in metres (T >= 5). ``fps`` overrides the default."""
        traj = np.asarray(trajectory, dtype=np.float64)
        if traj.ndim != 3 or traj.shape[1:] != (21, 3):
            raise ValueError(f"Trajectory must have shape (T, 21, 3). Got {traj.shape}")
        if traj.shape[0] < FeatureAggregator.MIN_FRAMES:
            raise ValueError(f"Trajectory needs at least {FeatureAggregator.MIN_FRAMES} frames. Got {traj.shape[0]}")
        if not np.all(np.isfinite(traj)):
            raise ValueError("Trajectory contains NaN or inf")

        features = self.aggregator.extract_features(self._fit_window(traj), fps=fps or self.fps)
        comb_vec = features["combined"].reshape(1, -1)
        predicted_label = str(self.model.predict(comb_vec)[0])

        probs: Dict[str, float] = {}
        confidence = 1.0
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(comb_vec)[0]
            probs = {str(c): float(p) for c, p in zip(self.model.classes_, proba)}
            confidence = float(np.max(proba))

        d = features["dynamics"]
        telemetry = {
            "peak_speed_m_s": float(d[0]), "mean_speed_m_s": float(d[1]),
            "peak_accel_m_s2": float(d[7]), "peak_jerk_m_s3": float(d[9]),
            "peak_ke_joules": float(d[11]), "total_energy_j_s": float(d[13]), "peak_force_n": float(d[14]),
        }
        return GesturePrediction(predicted_label, confidence, probs, telemetry)
