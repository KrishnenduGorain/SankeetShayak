"""Sign Gesture Inference Engine.

Provides fast, production-ready inference given sliding-window hand trajectories,
with input reliability checks, thread-safe loading, and clean telemetry mapping.
"""

from dataclasses import dataclass, field
import logging
from pathlib import Path
import threading
from typing import Dict, List, Optional
import warnings
import joblib
import numpy as np

from src.data.aggregator import FeatureAggregator

logger = logging.getLogger(__name__)

# Dynamics feature vector indices mapping
DYNAMICS_INDEX = {
    "peak_speed_m_s": 0,
    "mean_speed_m_s": 1,
    "peak_accel_m_s2": 7,
    "peak_jerk_m_s3": 9,
    "peak_ke_joules": 11,
    "total_energy_j_s": 13,
    "peak_force_n": 14,
}


class ModelNotFoundError(FileNotFoundError):
    """Raised when the trained gesture model file cannot be located."""
    pass


@dataclass
class GesturePrediction:
    """Prediction result and accompanying physical telemetry."""
    label: str
    confidence: float
    probabilities: Dict[str, float]
    telemetry: Dict[str, float]
    is_reliable: bool = True
    warnings: List[str] = field(default_factory=list)


class GesturePredictor:
    """Predicts sign labels from sliding window coordinate trajectories."""

    def __init__(self, model_path: Optional[str] = None, fps: float = 30.0, allow_autobuild: bool = False):
        self.fps = fps
        self.aggregator = FeatureAggregator(fps=fps)

        if model_path is None:
            repo_root = Path(__file__).resolve().parent.parent.parent
            default_path = repo_root / "models" / "handshayak_model.joblib"
            model_path = str(default_path)

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            if allow_autobuild:
                from src.models.train import build_and_train_model
                logger.info(f"Model missing at {self.model_path}. Auto-building trained model...")
                self.model = build_and_train_model(output_model_path=str(self.model_path))
            else:
                raise ModelNotFoundError(
                    f"Trained model not found at {self.model_path}. "
                    "Run 'python -m src.models.train' to build model weights."
                )
        else:
            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.simplefilter("always")
                self.model = joblib.load(self.model_path)
                version_warns = [str(w.message) for w in caught_warnings if "version" in str(w.message).lower()]
                if version_warns:
                    logger.warning(f"Scikit-learn model loaded with version notice: {version_warns[0]}")

    @property
    def classes(self) -> List[str]:
        """List of class labels learned by the model."""
        if hasattr(self.model, "classes_"):
            return [str(c) for c in self.model.classes_]
        return []

    def predict_trajectory(
        self,
        trajectory: np.ndarray,
        fps: Optional[float] = None,
    ) -> GesturePrediction:
        """Run inference on a gesture trajectory with reliability checks.

        Args:
            trajectory: Array of shape (T, 21, 3) normalized hand landmarks.
            fps: Optional sampling rate override.

        Returns:
            GesturePrediction containing predicted sign, confidence, telemetry, and reliability.
        """
        traj = np.asarray(trajectory, dtype=np.float64)
        if traj.ndim != 3 or traj.shape[1] != 21:
            raise ValueError(f"Trajectory must have shape (T, 21, 3). Got {traj.shape}")

        if not np.all(np.isfinite(traj)):
            raise ValueError("Trajectory contains NaN or infinite coordinates")

        active_fps = fps if (fps is not None and fps > 0) else self.fps
        warns: List[str] = []
        is_reliable = True

        # Extract combined features
        features = self.aggregator.extract_features(traj, fps=active_fps)
        comb_vec = features["combined"].reshape(1, -1)

        # Extract telemetry summary
        dyn_vec = features["dynamics"]
        telemetry = {k: float(dyn_vec[idx]) for k, idx in DYNAMICS_INDEX.items()}

        # 1. Stationary input check
        mean_speed = telemetry.get("mean_speed_m_s", 0.0)
        wrist_disp = np.linalg.norm(traj[-1, 0] - traj[0, 0])
        if mean_speed < 0.02 and wrist_disp < 0.02:
            is_reliable = False
            warns.append("Hand motion is stationary; dynamic gesture recognition confidence is reduced.")

        # 2. Absurd physiological velocity check (> 15 m/s)
        peak_speed = telemetry.get("peak_speed_m_s", 0.0)
        if peak_speed > 15.0:
            is_reliable = False
            warns.append(f"Unrealistic peak speed detected ({peak_speed:.1f} m/s); possible tracking artifact.")

        # Predict label & probabilities
        predicted_label = str(self.model.predict(comb_vec)[0])
        probs: Dict[str, float] = {}
        confidence = 1.0

        if hasattr(self.model, "predict_proba"):
            proba_arr = self.model.predict_proba(comb_vec)[0]
            classes = self.classes
            probs = {str(c): float(p) for c, p in zip(classes, proba_arr)}
            confidence = float(np.max(proba_arr))

        return GesturePrediction(
            label=predicted_label,
            confidence=confidence,
            probabilities=probs,
            telemetry=telemetry,
            is_reliable=is_reliable,
            warnings=warns,
        )


# Thread-safe global predictor singleton
_shared_predictor: Optional[GesturePredictor] = None
_predictor_lock = threading.Lock()


def get_shared_predictor(allow_autobuild: bool = False) -> GesturePredictor:
    """Thread-safe getter for shared GesturePredictor instance."""
    global _shared_predictor
    if _shared_predictor is None:
        with _predictor_lock:
            if _shared_predictor is None:
                _shared_predictor = GesturePredictor(allow_autobuild=allow_autobuild)
    return _shared_predictor
