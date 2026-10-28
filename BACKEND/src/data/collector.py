"""Gesture Data Collector and Synthetic Gesture Generator.

Authored for the Data Engineering Workstream (Lead: Koushik).
Enables recording labeled gesture sequences (e.g. gentle vs. forceful tap)
and provides synthetic gesture generation for continuous offline development.
"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

from src.data.asl_signs import LETTERS, NUMBERS, COMMON_SIGNS, BASE, get_signs


@dataclass
class RecordedGesture:
    """Represents a recorded sign gesture instance."""
    label: str
    subject_id: str
    fps: float
    # Normalized 3D trajectory array shape (T, 21, 3)
    trajectory: List[List[List[float]]]
    timestamp: str


class GestureDatasetBuilder:
    """Manages recording, saving, and generating sign language datasets."""

    def __init__(self, output_dir: str = "data/processed"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_sample(
        self,
        label: str,
        trajectory: np.ndarray,
        subject_id: str = "p01",
        fps: float = 30.0,
        filename: Optional[str] = None
    ) -> Path:
        """Save a single gesture trajectory to disk as JSON.

        Args:
            label: Sign label (e.g., 'A', 'B', 'HELLO').
            trajectory: Array of shape (T, 21, 3).
            subject_id: Identifier of the person performing the sign.
            fps: Frame rate of recording.
            filename: Optional output filename.

        Returns:
            Path to saved JSON file.
        """
        if filename is None:
            import time
            filename = f"{label}_{subject_id}_{int(time.time() * 1000)}.json"

        file_path = self.output_dir / filename
        data = {
            "label": label,
            "subject_id": subject_id,
            "fps": fps,
            "trajectory": trajectory.tolist(),
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        return file_path

    @staticmethod
    def _build_handshape(sign_dict):
        """Build full 21-landmark array from fingertip dict."""
        traj = [list(p) for p in BASE]
        # Tip indices
        tips = {4, 8, 12, 16, 20}
        for k, v in sign_dict.items():
            idx = int(k)
            traj[idx] = list(v)
        # Interpolate intermediate joints
        # Finger joint chains
        chains = [
            [5, 6, 7, 8],    # index
            [9, 10, 11, 12], # middle
            [13, 14, 15, 16],# ring
            [17, 18, 19, 20],# pinky
        ]
        for chain in chains:
            palm = traj[chain[0]]
            tip = traj[chain[-1]]
            for i in range(1, len(chain)):
                t = i / len(chain)
                traj[chain[i]] = [palm[j] + (tip[j] - palm[j]) * t for j in range(3)]
        # Thumb chain
        thumb_chain = [1, 2, 3, 4]
        palm = traj[1]
        tip = traj[4]
        for i in range(1, 4):
            t = [0.35, 0.55, 0.75][i - 1]
            traj[thumb_chain[i]] = [palm[j] + (tip[j] - palm[j]) * t for j in range(3)]
        return np.array(traj, dtype=np.float64)

    @staticmethod
    def generate_synthetic_gestures(
        samples_per_class: int = 20,
        fps: float = 30.0,
        frames_per_sample: int = 30
    ) -> Tuple[List[np.ndarray], np.ndarray, List[str]]:
        """Generate ASL sign language handshapes for training and testing.

        Covers:
        - ASL Alphabet: A-Z (26 classes)
        - ASL Numbers: 0-9 (10 classes)
        - Common Signs: HELLO, YES, NO, THANK_YOU, PLEASE, SORRY, I_LOVE_YOU, HELP, STOP, MORE, EAT, DRINK (12 classes)

        Total: 48 distinct sign classes.

        Returns:
            Tuple of (trajectories: List of arrays (T, 21, 3), labels: array (N,), class_names)
        """
        classes = get_signs()
        trajectories = []
        labels = []

        dt = 1.0 / fps
        t = np.linspace(0, (frames_per_sample - 1) * dt, frames_per_sample)

        # Combine all sign dictionaries
        all_signs = {}
        all_signs.update(LETTERS)
        all_signs.update(NUMBERS)
        all_signs.update(COMMON_SIGNS)

        for class_name in classes:
            sign_dict = all_signs.get(class_name, LETTERS['A'])  # fallback to A
            base_shape = GestureDatasetBuilder._build_handshape(sign_dict)

            for _ in range(samples_per_class):
                # Tile base shape across frames
                sample_traj = np.tile(base_shape, (frames_per_sample, 1, 1))

                # Add small temporal variation (slight wobble)
                noise = np.random.normal(0, 0.003, size=sample_traj.shape)
                sample_traj += noise

                # Add slight frame-to-frame variation for realism
                phase = np.random.random() * 2 * np.pi
                wobble = np.sin(np.linspace(phase, phase + 2 * np.pi, frames_per_sample))[:, None, None] * 0.002
                sample_traj += wobble * np.random.normal(0, 1, (frames_per_sample, 21, 3))

                trajectories.append(sample_traj)
                labels.append(class_name)

        return trajectories, np.array(labels), classes
