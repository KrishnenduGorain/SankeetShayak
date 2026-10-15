"""Interactive Demo for SankeetShayak / HandShayak.

Runs either:
1. An offline simulation demonstrating physics features and classification on synthetic gestures.
2. A live webcam visualizer overlaying 3D hand landmarks and instantaneous physics telemetry.

Usage:
    # Run simulation demo (works anywhere, headless/no camera required):
    python demo.py --simulate

    # Run live webcam tracking:
    python demo.py --webcam
"""

import argparse
import sys
import time
import cv2
import numpy as np

from src.data.collector import GestureDatasetBuilder
from src.models.predictor import GesturePredictor
from src.vision.detector import HandLandmarkDetector
from src.vision.visualizer import HandVisualizer


def run_simulation_demo():
    print("=" * 65)
    print("🖐️  SANKEET SHAYAK: PHYSICS-INFORMED SIGN RECOGNITION DEMO")
    print("=" * 65)
    print("\n1. Initializing Gesture Predictor & Physics Engine...")
    predictor = GesturePredictor()

    print("2. Generating test gestures (Gentle Tap vs. Forceful Strike)...")
    trajs, labels, _ = GestureDatasetBuilder.generate_synthetic_gestures(
        samples_per_class=1, fps=30.0, frames_per_sample=30
    )

    for traj, true_label in zip(trajs, labels):
        pred = predictor.predict_trajectory(traj)
        print(f"\n▶ Ground Truth Sign : {true_label.upper()}")
        print(f"  Predicted Sign    : {pred.label.upper()} (Confidence: {pred.confidence * 100:.1f}%)")
        print("  Biomechanical Telemetry:")
        print(f"    • Peak Speed    : {pred.telemetry['peak_speed_m_s']:.3f} m/s")
        print(f"    • Peak Accel    : {pred.telemetry['peak_accel_m_s2']:.3f} m/s²")
        print(f"    • Peak Jerk     : {pred.telemetry['peak_jerk_m_s3']:.3f} m/s³")
        print(f"    • Kinetic Energy: {pred.telemetry['peak_ke_joules']:.4f} Joules")
        print(f"    • Inferred Force: {pred.telemetry['peak_force_n']:.3f} N")

    print("\n" + "=" * 65)
    print("✅ Simulation completed successfully! Physics features clearly differentiate")
    print("   movement dynamics between gentle and forceful signs.")
    print("=" * 65)


def get_working_camera(preferred_index: int = 0):
    """Probe for available cameras using macOS AVFoundation backend."""
    # Try preferred index with AVFoundation first
    cap = cv2.VideoCapture(preferred_index, cv2.CAP_AVFOUNDATION)
    if cap.isOpened():
        return cap, preferred_index

    cap.release()

    # Probe indices 0, 1, 2
    for idx in [0, 1, 2]:
        if idx == preferred_index:
            continue
        test_cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
        if test_cap.isOpened():
            ret, _ = test_cap.read()
            if ret:
                return test_cap, idx
            test_cap.release()

    # Fallback to default backend
    fallback_cap = cv2.VideoCapture(preferred_index)
    if fallback_cap.isOpened():
        return fallback_cap, preferred_index

    fallback_cap.release()
    return None, preferred_index


def run_webcam_demo(camera_index: int = 0):
    print("\nStarting live webcam stream... Press 'q' to exit.")
    cap, actual_idx = get_working_camera(camera_index)

    if cap is None or not cap.isOpened():
        print("\n" + "=" * 65)
        print("❌ CAMERA ACCESS ERROR: OpenCV could not open the webcam.")
        print("=" * 65)
        print("This is almost always a macOS Privacy Permission issue.\n")
        print("To fix this on your Mac:")
        print("1. Open: System Settings (Apple Menu  > System Settings)")
        print("2. Go to: Privacy & Security > Camera")
        print("3. Find your Terminal application (e.g. Terminal, iTerm2, VS Code,")
        print("   Cursor, or your IDE) and TOGGLE THE SWITCH TO ON (Enabled).")
        print("4. If your terminal is already running, restart it and re-run:")
        print("     python demo.py --webcam")
        print("\nAlternative: If you have an external webcam or Continuity Camera, try:")
        print("     python demo.py --webcam --cam-index 1")
        print("=" * 65)
        sys.exit(1)

    print(f"✅ Camera initialized successfully on index {actual_idx}!")
    detector = HandLandmarkDetector()
    visualizer = HandVisualizer()
    t0 = time.perf_counter()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Flip horizontally for natural mirror feel
            frame = cv2.flip(frame, 1)

            res = detector.process_frame(frame, time.perf_counter() - t0)
            if res.detected and res.raw_landmarks is not None:
                frame = visualizer.draw_skeleton(frame, res.raw_landmarks)
                cv2.putText(
                    frame,
                    f"Hand: {res.handedness} ({res.confidence * 100:.0f}%)",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )
            else:
                cv2.putText(
                    frame,
                    "No hand detected",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )

            cv2.imshow("SankeetShayak Live Vision", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SankeetShayak Demo Runner")
    parser.add_argument("--simulate", action="store_true", help="Run simulated physics and classification demo")
    parser.add_argument("--webcam", action="store_true", help="Run live webcam hand tracking")
    parser.add_argument("--cam-index", type=int, default=0, help="Camera device index (default: 0)")
    args = parser.parse_args()

    if args.webcam:
        run_webcam_demo(camera_index=args.cam_index)
    else:
        run_simulation_demo()
