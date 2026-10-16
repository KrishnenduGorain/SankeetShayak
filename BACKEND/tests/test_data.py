"""Unit tests for Feature Aggregation and Fusion (src/data)."""

import numpy as np
import pytest
from src.data.aggregator import FeatureAggregator


def test_feature_aggregator_output_shapes():
    """Verify aggregator outputs shape, dynamics, and combined vectors properly."""
    fps = 30.0
    T = 25
    aggregator = FeatureAggregator(fps=fps)

    # Synthetic normalized trajectory: (T, 21, 3)
    traj = np.random.normal(loc=0.0, scale=0.2, size=(T, 21, 3))
    # Anchor wrist to 0
    traj[:, 0, :] = 0.0

    features = aggregator.extract_features(traj, fps=fps)

    assert "shape" in features
    assert "dynamics" in features
    assert "combined" in features

    shape_vec = features["shape"]
    dyn_vec = features["dynamics"]
    comb_vec = features["combined"]

    assert shape_vec.ndim == 1
    assert dyn_vec.ndim == 1
    assert comb_vec.ndim == 1

    # Combined should be exact concatenation of shape and dynamics
    assert len(comb_vec) == len(shape_vec) + len(dyn_vec)
    np.testing.assert_array_equal(comb_vec, np.concatenate([shape_vec, dyn_vec]))
    # No NaNs or Infs
    assert np.all(np.isfinite(comb_vec))
