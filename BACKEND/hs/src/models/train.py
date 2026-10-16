"""Model Training Script for HandShayak.

Authored for the ML & Backend Workstream (Lead: Harsimran).
Trains the combined physics-informed classifier, runs the participant-independent ablation study,
and saves the production model (atomically) plus a small metadata file.

    python -m src.models.train --samples 30
"""

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Optional
import joblib
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.aggregator import FeatureAggregator
from src.data.collector import GestureDatasetBuilder
from src.models.ablation import AblationStudy


def default_model_path() -> Path:
    from src.config import settings
    return settings.model_path


def build_and_train_model(
    samples_per_class: int = 30,
    output_model_path: Optional[str] = None,
    run_ablation: bool = True,
    verbose: bool = True,
    seed: int = 42,
) -> Pipeline:
    """Build dataset, run ablation benchmarking, and save the production model."""
    log = print if verbose else (lambda *a, **k: None)
    out_path = Path(output_model_path) if output_model_path else default_model_path()

    log("1. Generating / loading gesture trajectories...")
    ds = GestureDatasetBuilder.generate_synthetic_dataset(samples_per_class=samples_per_class, fps=30.0,
                                                          frames_per_sample=30, seed=seed)
    aggregator = FeatureAggregator(fps=30.0)

    log("2. Extracting fused biomechanical and shape features...")
    feats = [aggregator.extract_features(t, fps=30.0) for t in ds.trajectories]
    X_shape = np.array([f["shape"] for f in feats])
    X_dyn = np.array([f["dynamics"] for f in feats])
    X_comb = np.array([f["combined"] for f in feats])
    y = np.array(ds.labels)
    groups = np.array(ds.groups)

    if run_ablation:
        log("\n3. Running ablation study (participant-independent split)...")
        results = AblationStudy(random_state=seed).run_study(X_shape, X_dyn, X_comb, y, groups=groups)
        for name, res in results.items():
            log(f"   [{name}] Accuracy: {res.accuracy * 100:.1f}% | Macro-F1: {res.macro_f1:.3f} | Features: {res.feature_dim}")

    log("\n4. Training final production pipeline...")
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=seed)),
    ])
    pipeline.fit(X_comb, y)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=out_path.parent, suffix=".tmp")
    os.close(fd)
    try:
        joblib.dump(pipeline, tmp)
        os.replace(tmp, out_path)                       # atomic: concurrent readers never see a partial file
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    meta = {"classes": [str(c) for c in pipeline.classes_], "feature_dim": int(X_comb.shape[1]),
            "sklearn": sklearn.__version__, "samples_per_class": samples_per_class, "window_frames": 30, "fps": 30.0}
    out_path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log(f"5. Saved production model to {out_path}")
    return pipeline


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train the HandShayak classifier")
    ap.add_argument("--samples", type=int, default=30, help="synthetic samples per class")
    ap.add_argument("--output", default=None, help="output .joblib path")
    ap.add_argument("--no-ablation", action="store_true")
    args = ap.parse_args()
    build_and_train_model(args.samples, args.output, run_ablation=not args.no_ablation)
