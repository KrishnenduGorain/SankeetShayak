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
            label: Sign label (e.g., 'tap_gentle', 'tap_forceful').
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
    def generate_synthetic_gestures(
        samples_per_class: int = 20,
        fps: float = 30.0,
        frames_per_sample: int = 30
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Generate physically distinct synthetic gestures for training and testing.

        Simulates 4 benchmark classes:
        1. 'tap_gentle': Slow downward tap (low velocity, low kinetic energy)
        2. 'tap_forceful': Fast abrupt strike (high velocity, high acceleration, high jerk)
        3. 'swipe_slow': Steady horizontal translation (medium velocity, low jerk)
        4. 'swipe_rapid': Rapid flick (high velocity, high jerk)

        Returns:
            Tuple of (trajectories: List of arrays (T, 21, 3), labels: array (N,), class_names)
        """
        classes = ["tap_gentle", "tap_forceful", "swipe_slow", "swipe_rapid"]
        trajectories = []
        labels = []

        dt = 1.0 / fps
        t = np.linspace(0, (frames_per_sample - 1) * dt, frames_per_sample)

        for class_idx, class_name in enumerate(classes):
            for _ in range(samples_per_class):
                # Base resting hand: 21 landmarks centered at wrist (0, 0, 0)
                base_hand = np.zeros((21, 3), dtype=np.float64)
                # Spread out finger joints naturally
                for finger in range(5):
                    for joint in range(1, 5):
                        idx = finger * 4 + joint
                        base_hand[idx] = [(finger - 2) * 0.15, joint * 0.2, 0.0]

                sample_traj = np.tile(base_hand, (frames_per_sample, 1, 1))

                # Add noise
                noise = np.random.normal(0, 0.005, size=sample_traj.shape)
                sample_traj += noise

                if class_name == "tap_gentle":
                    # Gentle tap: smooth sine wave with low peak velocity
                    vy = -0.3 * np.sin(np.pi * t / t[-1])
                    sample_traj[:, :, 1] += vy[:, None]
                elif class_name == "tap_forceful":
                    # Forceful strike: sharp quadratic descent, abrupt stop (high jerk)
                    vy = -1.2 * np.sin(np.pi * t / t[-1]) ** 3
                    sample_traj[:, :, 1] += vy[:, None]
                elif class_name == "swipe_slow":
                    # Steady slow horizontal swipe
                    vx = 0.4 * (t / t[-1])
                    sample_traj[:, :, 0] += vx[:, None]
                elif class_name == "swipe_rapid":
                    # Rapid flick: fast exponential swipe
                    vx = 1.5 * (t / t[-1]) ** 2
                    sample_traj[:, :, 0] += vx[:, None]

                trajectories.append(sample_traj)
                labels.append(class_name)

        return trajectories, np.array(labels), classes
