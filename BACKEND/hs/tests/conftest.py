import os
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HANDSHAYAK_OFFLINE", "0")


@pytest.fixture(scope="session")
def model_path(tmp_path_factory):
    """Train a small model once per session into a temp dir and point the app at it."""
    from src.config import settings
    from src.models.train import build_and_train_model

    path = tmp_path_factory.mktemp("models") / "test_model.joblib"
    build_and_train_model(samples_per_class=24, output_model_path=str(path), run_ablation=False, verbose=False)
    old = settings.MODEL_PATH
    settings.MODEL_PATH = str(path)
    yield path
    settings.MODEL_PATH = old


@pytest.fixture(scope="session")
def dataset():
    from src.data.collector import GestureDatasetBuilder
    return GestureDatasetBuilder.generate_synthetic_dataset(samples_per_class=12, participants=6, seed=1)
