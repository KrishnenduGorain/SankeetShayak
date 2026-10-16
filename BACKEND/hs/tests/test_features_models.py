import numpy as np
import pytest

from src.data.aggregator import DYNAMICS_FEATURE_NAMES, SHAPE_FEATURE_NAMES, FeatureAggregator
from src.data.collector import CLASS_NAMES, GestureDatasetBuilder
from src.models.ablation import AblationStudy
from src.models.predictor import GesturePredictor


def test_collector_shapes_and_api_compat():
    trajs, labels, names = GestureDatasetBuilder.generate_synthetic_gestures(samples_per_class=2)
    assert len(trajs) == len(labels) == 2 * len(CLASS_NAMES) and names == CLASS_NAMES
    assert all(t.shape == (30, 21, 3) and np.isfinite(t).all() for t in trajs)


def test_collector_is_deterministic_and_validates():
    a = GestureDatasetBuilder.generate_synthetic_dataset(2, seed=5).trajectories[3]
    b = GestureDatasetBuilder.generate_synthetic_dataset(2, seed=5).trajectories[3]
    assert np.array_equal(a, b)
    with pytest.raises(ValueError):
        GestureDatasetBuilder.generate_synthetic_dataset(0)


def test_feature_layout_matches_predictor_indices(dataset):
    f = FeatureAggregator(30.0).extract_features(dataset.trajectories[0])
    assert f["shape"].shape == (len(SHAPE_FEATURE_NAMES),) and f["dynamics"].shape == (len(DYNAMICS_FEATURE_NAMES),)
    assert f["combined"].shape[0] == f["shape"].shape[0] + f["dynamics"].shape[0]
    names = DYNAMICS_FEATURE_NAMES
    assert (names[0], names[1], names[7], names[9], names[11], names[13], names[14]) == (
        "peak_speed", "mean_speed", "peak_accel", "peak_jerk", "peak_ke", "total_energy", "peak_force")
    assert np.isfinite(f["combined"]).all()


def test_dynamics_scale_with_speed(dataset):
    ag = FeatureAggregator(30.0)
    def mean_feat(label, idx):
        return np.mean([ag.extract_features(t)["dynamics"][idx] for t, l in zip(dataset.trajectories, dataset.labels) if l == label])
    assert mean_feat("tap_forceful", 0) > 2.0 * mean_feat("tap_gentle", 0)
    assert mean_feat("tap_forceful", 11) > 8.0 * mean_feat("tap_gentle", 11)       # KE ~ v^2
    assert mean_feat("idle", 0) < 0.15


def test_aggregator_rejects_bad_input():
    ag = FeatureAggregator()
    for bad in (np.zeros((3, 21, 3)), np.zeros((10, 20, 3)), np.full((10, 21, 3), np.inf)):
        with pytest.raises(ValueError):
            ag.extract_features(bad)


def test_ablation_is_participant_independent_and_dynamics_help(dataset):
    ag = FeatureAggregator(30.0)
    feats = [ag.extract_features(t) for t in dataset.trajectories]
    X = {k: np.array([f[k] for f in feats]) for k in ("shape", "dynamics", "combined")}
    res = AblationStudy(42).run_study(X["shape"], X["dynamics"], X["combined"], np.array(dataset.labels), groups=np.array(dataset.groups))
    shape, dyn, comb = res["Shape-Only Baseline"], res["Dynamics-Only Baseline"], res["HandShayak (Combined)"]
    assert comb.participant_independent
    assert comb.accuracy >= shape.accuracy + 0.2          # shape alone cannot tell gentle from forceful
    assert comb.accuracy >= 0.9 and dyn.accuracy > shape.accuracy


def test_ablation_handles_tiny_and_mismatched_data():
    X = np.random.rand(6, 4); y = np.array(list("aabbcc"))
    assert AblationStudy().run_study(X, X, X, y)["Shape-Only Baseline"].feature_dim == 4
    with pytest.raises(ValueError):
        AblationStudy().run_study(X, X[:3], X, y)


def test_predictor_end_to_end(model_path, dataset):
    p = GesturePredictor(str(model_path))
    ok = 0
    sample = [(t, l) for t, l in zip(dataset.trajectories, dataset.labels)][:20]
    for t, l in sample:
        ok += p.predict_trajectory(t).label == l
    assert ok / len(sample) >= 0.85
    pred = p.predict_trajectory(dataset.trajectories[0])
    assert abs(sum(pred.probabilities.values()) - 1) < 1e-6 and set(pred.telemetry) >= {"peak_speed_m_s", "peak_force_n"}


def test_predictor_window_handling_and_errors(model_path, dataset):
    p = GesturePredictor(str(model_path))
    t = dataset.trajectories[5]
    assert p.predict_trajectory(t[-12:]).label in CLASS_NAMES          # short window is padded
    assert p.predict_trajectory(np.concatenate([t, t])).label in CLASS_NAMES   # long window trimmed
    for bad in (np.zeros((4, 21, 3)), np.zeros((20, 20, 3)), np.full((20, 21, 3), np.nan)):
        with pytest.raises(ValueError):
            p.predict_trajectory(bad)


def test_predictor_recovers_from_corrupt_model(tmp_path):
    bad = tmp_path / "m.joblib"
    bad.write_bytes(b"not a model")
    p = GesturePredictor(str(bad))
    assert p.predict_trajectory(GestureDatasetBuilder.generate_synthetic_dataset(1).trajectories[1]).label in CLASS_NAMES
