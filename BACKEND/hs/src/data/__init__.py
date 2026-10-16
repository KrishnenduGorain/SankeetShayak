"""Data module: synthetic gesture generation and feature aggregation."""

from src.data.aggregator import DYNAMICS_FEATURE_NAMES, SHAPE_FEATURE_NAMES, FeatureAggregator
from src.data.collector import CLASS_NAMES, GestureDatasetBuilder, SyntheticDataset

__all__ = ["FeatureAggregator", "DYNAMICS_FEATURE_NAMES", "SHAPE_FEATURE_NAMES",
           "GestureDatasetBuilder", "SyntheticDataset", "CLASS_NAMES"]
