"""Vision utility: Hand Landmark, Skeletal, and Physics Telemetry Visualizer.

Renders 21 3D hand joints, connecting rigid-body linkages, velocity vectors,
and real-time biomechanical HUD overlays (speed, accel, jerk, kinetic energy, force)
onto OpenCV frames across all operating systems.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import cv2
from src.vision.detector import HAND_CONNECTIONS


class HandVisualizer:
    """Draws hand skeletons and kinematic/dynamic HUD overlays onto video frames."""

    def __init__(self):
        self.connections = HAND_CONNECTIONS

    def draw_skeleton(
        self,
        frame_bgr: np.ndarray,
        raw_landmarks: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Draw bone linkages and joint landmarks on a frame.

        Args:
            frame_bgr: BGR frame array (H, W, 3).
            raw_landmarks: Optional (21, 3) normalized [0, 1] landmark coordinates.

        Returns:
            Annotated frame array.
        """
        output = frame_bgr.copy()
        if raw_landmarks is None or len(raw_landmarks) != 21:
            return output

        h, w = output.shape[:2]

        # Draw bone connections
        for start_idx, end_idx in self.connections:
            p1 = (int(raw_landmarks[start_idx, 0] * w), int(raw_landmarks[start_idx, 1] * h))
            p2 = (int(raw_landmarks[end_idx, 0] * w), int(raw_landmarks[end_idx, 1] * h))
            cv2.line(output, p1, p2, (0, 255, 128), 2, cv2.LINE_AA)

        # Draw joint nodes
        for idx in range(21):
            px = int(raw_landmarks[idx, 0] * w)
            py = int(raw_landmarks[idx, 1] * h)
            # Differentiate fingertips with orange circle
            color = (0, 140, 255) if idx in (4, 8, 12, 16, 20) else (255, 200, 0)
            radius = 5 if idx in (4, 8, 12, 16, 20) else 3
            cv2.circle(output, (px, py), radius, color, -1, cv2.LINE_AA)

        return output

    def draw_velocity_vectors(
        self,
        frame_bgr: np.ndarray,
        raw_landmarks: np.ndarray,
        velocities_21x3: np.ndarray,
        scale: float = 25.0
    ) -> np.ndarray:
        """Draw dynamic 2D velocity vector arrows protruding from joints.

        Args:
            frame_bgr: BGR frame array (H, W, 3).
            raw_landmarks: Array of shape (21, 3) normalized coords.
            velocities_21x3: Array of shape (21, 3) velocity vectors (m/s).
            scale: Pixel scale factor for arrow length.

        Returns:
            Annotated frame array.
        """
        output = frame_bgr.copy()
        h, w = output.shape[:2]

        # Draw velocity vectors for fingertips (indices 4, 8, 12, 16, 20)
        fingertip_indices = [4, 8, 12, 16, 20]
        for tip in fingertip_indices:
            px = int(raw_landmarks[tip, 0] * w)
            py = int(raw_landmarks[tip, 1] * h)

            vx = velocities_21x3[tip, 0]
            vy = velocities_21x3[tip, 1]

            arrow_end = (int(px + vx * scale), int(py + vy * scale))
            speed = np.sqrt(vx ** 2 + vy ** 2)

            if speed > 0.05:  # Only draw when moving
                cv2.arrowedLine(output, (px, py), arrow_end, (0, 255, 255), 2, tipLength=0.3)

        return output

    def draw_telemetry_hud(
        self,
        frame_bgr: np.ndarray,
        telemetry: Dict[str, float],
        prediction_label: Optional[str] = None,
        confidence: Optional[float] = None,
    ) -> np.ndarray:
        """Draw a semi-transparent biomechanical telemetry HUD on the frame.

        Args:
            frame_bgr: BGR frame array (H, W, 3).
            telemetry: Dict with keys: speed, accel, jerk, ke, force.
            prediction_label: Optional recognized sign label.
            confidence: Optional confidence score [0, 1].

        Returns:
            Annotated frame array.
        """
        output = frame_bgr.copy()
        h, w = output.shape[:2]

        # Define HUD card dimensions
        card_w, card_h = 300, 210
        card_x, card_y = 15, 15

        # Create semi-transparent overlay
        overlay = output.copy()
        cv2.rectangle(overlay, (card_x, card_y), (card_x + card_w, card_y + card_h), (20, 20, 20), -1)
        cv2.rectangle(output, (card_x, card_y), (card_x + card_w, card_y + card_h), (0, 255, 128), 1)
        alpha = 0.75
        cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

        # Draw HUD Header
        cv2.putText(output, "SANKEET SHAYAK TELEMETRY", (card_x + 12, card_y + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 128), 2, cv2.LINE_AA)

        # Draw Prediction Badge if available
        if prediction_label:
            conf_str = f" ({confidence * 100:.0f}%)" if confidence is not None else ""
            pred_text = f"SIGN: {prediction_label.upper()}{conf_str}"
            cv2.putText(output, pred_text, (card_x + 12, card_y + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(output, "STATUS: Tracking Active", (card_x + 12, card_y + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1, cv2.LINE_AA)

        # Telemetry metrics list
        speed = telemetry.get("speed_m_s", 0.0)
        accel = telemetry.get("accel_m_s2", 0.0)
        jerk = telemetry.get("jerk_m_s3", 0.0)
        ke = telemetry.get("ke_joules", 0.0)
        force = telemetry.get("force_n", 0.0)

        metrics = [
            ("Speed", f"{speed:.2f} m/s", (255, 255, 255)),
            ("Accel", f"{accel:.2f} m/s2", (220, 220, 220)),
            ("Jerk", f"{jerk:.2f} m/s3", (220, 220, 220)),
            ("Energy", f"{ke:.4f} J", (0, 200, 255)),
            ("Force", f"{force:.2f} N", (0, 255, 180)),
        ]

        start_y = card_y + 75
        for idx, (label, val, col) in enumerate(metrics):
            y_pos = start_y + idx * 26
            cv2.putText(output, f"{label}:", (card_x + 12, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)
            cv2.putText(output, val, (card_x + 95, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, col, 1, cv2.LINE_AA)

            # Energy bar indicator
            if label == "Energy":
                bar_x = card_x + 195
                bar_w = 90
                bar_h = 10
                bar_fill = int(min(1.0, ke / 0.5) * bar_w)
                cv2.rectangle(output, (bar_x, y_pos - 9), (bar_x + bar_w, y_pos + 1), (60, 60, 60), -1)
                cv2.rectangle(output, (bar_x, y_pos - 9), (bar_x + bar_fill, y_pos + 1), (0, 200, 255), -1)

        return output
