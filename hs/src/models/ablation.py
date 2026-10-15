"""Ablation Study Framework.

Authored for the ML & Backend Architecture Workstream (Lead: Harsimran).
Compares three classifier variants on IDENTICAL splits:
1. Shape-Only Baseline      (visual geometry only)
2. Dynamics-Only Baseline   (kinematic & energy metrics only)
3. HandShayak Combined      (fused shape + dynamics)

When participant ids (``groups``) are supplied the split is participant-independent: no signer appears
in both train and test, as required by the project report (an i.i.d. split lets the model memorise a
person's habits and overstates accuracy).
"""

from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
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
    participant_independent: bool = False


class AblationStudy:
    """Executes objective ablation comparison across feature spaces."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def _split(self, y: np.ndarray, groups: Optional[np.ndarray], test_size: float):
        idx = np.arange(len(y))
        if groups is not None and len(np.unique(groups)) >= 2:
            # try a few seeds so that every class is present in the training part
            for k in range(20):
                gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=self.random_state + k)
                tr, te = next(gss.split(idx, y, groups))
                if set(np.unique(y)) <= set(np.unique(y[tr])) and len(te) > 0:
                    return tr, te, True
        try:
            tr, te = train_test_split(idx, test_size=test_size, random_state=self.random_state, stratify=y)
        except ValueError:                       # too few samples in some class for stratification
            tr, te = train_test_split(idx, test_size=test_size, random_state=self.random_state)
        return tr, te, False

    def run_study(
        self,
        X_shape: np.ndarray,
        X_dynamics: np.ndarray,
        X_combined: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.3,
        groups: Optional[np.ndarray] = None,
    ) -> Dict[str, AblationResult]:
        y = np.asarray(y)
        groups = None if groups is None else np.asarray(groups)
        if not (len(X_shape) == len(X_dynamics) == len(X_combined) == len(y)):
            raise ValueError("feature matrices and labels must have the same number of rows")
        idx_train, idx_test, independent = self._split(y, groups, test_size)
        y_train, y_test = y[idx_train], y[idx_test]

        conditions = {
            "Shape-Only Baseline": X_shape,
            "Dynamics-Only Baseline": X_dynamics,
            "HandShayak (Combined)": X_combined,
        }
        results: Dict[str, AblationResult] = {}
        for name, X in conditions.items():
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", RandomForestClassifier(n_estimators=100, random_state=self.random_state)),
            ])
            pipe.fit(X[idx_train], y_train)
            pred = pipe.predict(X[idx_test])
            results[name] = AblationResult(
                name=name,
                accuracy=float(accuracy_score(y_test, pred)),
                macro_f1=float(f1_score(y_test, pred, average="macro", zero_division=0)),
                feature_dim=int(X.shape[1]),
                classification_report=classification_report(y_test, pred, zero_division=0),
                participant_independent=independent,
            )
        return results
