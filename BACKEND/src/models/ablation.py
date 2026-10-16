"""Ablation Study Framework.

Authored for the ML & Backend Architecture Workstream (Lead: Harsimran).
Compares three classifier variants:
1. Shape-Only Baseline (Visual geometry only)
2. Dynamics-Only Baseline (Kinematic & energy metrics only)
3. HandShayak Combined (Fused Shape + Dynamics)
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class AblationResult:
    """Evaluation metrics for an ablation condition."""
    name: str
    accuracy: float
    macro_f1: float
    feature_dim: int
    classification_report: str


class AblationStudy:
    """Executes objective ablation comparison across feature spaces."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def run_study(
        self,
        X_shape: np.ndarray,
        X_dynamics: np.ndarray,
        X_combined: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.3
    ) -> Dict[str, AblationResult]:
        """Run 3-way ablation comparison.

        Args:
            X_shape: Feature matrix (N, D_shape).
            X_dynamics: Feature matrix (N, D_dynamics).
            X_combined: Feature matrix (N, D_combined).
            y: Target label array (N,).
            test_size: Ratio of test partition.

        Returns:
            Dictionary mapping condition name to AblationResult.
        """
        # Ensure identical train/test splits across conditions
        indices = np.arange(len(y))
        idx_train, idx_test = train_test_split(
            indices, test_size=test_size, random_state=self.random_state, stratify=y
        )
        y_train, y_test = y[idx_train], y[idx_test]

        conditions = {
            "Shape-Only Baseline": (X_shape[idx_train], X_shape[idx_test]),
            "Dynamics-Only Baseline": (X_dynamics[idx_train], X_dynamics[idx_test]),
            "HandShayak (Combined)": (X_combined[idx_train], X_combined[idx_test]),
        }

        results = {}

        for cond_name, (X_tr, X_te) in conditions.items():
            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", RandomForestClassifier(n_estimators=100, random_state=self.random_state))
            ])
            pipeline.fit(X_tr, y_train)
            y_pred = pipeline.predict(X_te)

            acc = float(accuracy_score(y_test, y_pred))
            f1 = float(f1_score(y_test, y_pred, average="macro"))
            report = classification_report(y_test, y_pred, zero_division=0)

            results[cond_name] = AblationResult(
                name=cond_name,
                accuracy=acc,
                macro_f1=f1,
                feature_dim=X_tr.shape[1],
                classification_report=report,
            )

        return results
