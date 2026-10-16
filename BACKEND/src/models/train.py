"""Model Training Script for SankeetShayak.

Authored for the ML & Backend Workstream (Lead: Harsimran).
Trains the combined physics-informed classifier, runs the ablation study,
and saves the production model artifact to models/handshayak_model.joblib.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.aggregator import FeatureAggregator
from src.data.collector import GestureDatasetBuilder
from src.models.ablation import AblationStudy


def build_and_train_model(
    samples_per_class: int = 30,
    output_model_path: str = "models/handshayak_model.joblib",
    run_ablation: bool = True
) -> Pipeline:
    """Build dataset, run ablation benchmarking, and save production model.

    Args:
        samples_per_class: Synthetic samples per gesture class if no real data is loaded.
        output_model_path: Output file path for saved joblib model.
        run_ablation: Whether to log ablation comparison results.

    Returns:
        Fitted Scikit-Learn Pipeline.
    """
    print("1. Generating / Loading gesture trajectories...")
    trajectories, labels, class_names = GestureDatasetBuilder.generate_synthetic_gestures(
        samples_per_class=samples_per_class, fps=30.0, frames_per_sample=30
    )

    aggregator = FeatureAggregator(fps=30.0)

    X_shape_list = []
    X_dynamics_list = []
    X_combined_list = []

    print("2. Extracting fused biomechanical and shape features...")
    for traj in trajectories:
        feat = aggregator.extract_features(traj, fps=30.0)
        X_shape_list.append(feat["shape"])
        X_dynamics_list.append(feat["dynamics"])
        X_combined_list.append(feat["combined"])

    X_shape = np.array(X_shape_list)
    X_dynamics = np.array(X_dynamics_list)
    X_combined = np.array(X_combined_list)
    y = np.array(labels)

    if run_ablation:
        print("\n3. Running Ablation Study...")
        ablation = AblationStudy()
        results = ablation.run_study(X_shape, X_dynamics, X_combined, y)
        for name, res in results.items():
            print(f"   [{name}] Accuracy: {res.accuracy * 100:.1f}% | Macro-F1: {res.macro_f1:.3f} | Features: {res.feature_dim}")

    print("\n4. Training Final Production Pipeline...")
    production_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    production_pipeline.fit(X_combined, y)

    # Save to disk
    out_path = Path(output_model_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(production_pipeline, out_path)
    print(f"5. Successfully saved production model to {out_path}!")

    return production_pipeline


if __name__ == "__main__":
    build_and_train_model()
