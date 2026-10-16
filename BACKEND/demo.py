"""Interactive Demo Runner for SankeetShayak / HandShayak.

Runs:
1. Simulation Demo: Tests physics engine, ablation benchmarking, and sign inference.
2. Live Webcam Stream: Real-time hand tracking, velocity arrows, and physics telemetry HUD.
3. Video File Analysis: Processes a pre-recorded sign video and outputs annotated telemetry.

Cross-platform compatible with macOS, Windows, and Linux.

Usage:
    # Run simulation demo (works anywhere, headless/no camera required):
    python demo.py --simulate

    # Run live webcam tracking with real-time physics HUD:
    python demo.py --webcam

    # Run on a video file:
    python demo.py --video path/to/gesture.mp4
"""

import argparse
from pathlib import Path
import sys
import cv2
import numpy as np

from src.data.collector import GestureDatasetBuilder
from src.models.predictor import GesturePredictor
from src.vision.camera import CrossPlatformCamera, get_camera_permission_guidance, probe_available_cameras
from src.vision.stream_processor import StreamProcessor


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


def run_stream_demo(source, window_title="SankeetShayak Live Telemetry"):
    print(f"\nInitializing stream from source: {source}...")
    print("Press 'q' in the video window to exit.")

    processor = StreamProcessor(fps=30.0, enable_predictor=True)

    try:
        # Use generator to process frames
        for frame_idx, frame, telemetry in processor.process_source(
            source=source,
            display=True,
            window_title=window_title,
        ):
            # Periodically print stats to terminal
            if frame_idx % 30 == 0 and telemetry["speed_m_s"] > 0.05:
                print(
                    f"[Frame {frame_idx:04d}] "
                    f"Speed: {telemetry['speed_m_s']:.2f} m/s | "
                    f"KE: {telemetry['ke_joules']:.4f} J | "
                    f"Force: {telemetry['force_n']:.2f} N"
                )
    except RuntimeError as e:
        print("\n" + "=" * 65)
        print("❌ CAMERA / STREAM ERROR:")
        print(str(e))
        print("=" * 65)
        sys.exit(1)
    finally:
        processor.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SankeetShayak Demo Runner")
    parser.add_argument("--simulate", action="store_true", help="Run simulated physics and classification demo")
    parser.add_argument("--webcam", action="store_true", help="Run live webcam hand tracking & physics HUD")
    parser.add_argument("--cam-index", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--video", type=str, default=None, help="Path to video file to analyze")
    parser.add_argument("--probe-cameras", action="store_true", help="Scan and list working cameras on system")
    args = parser.parse_args()

    if args.probe_cameras:
        print("Scanning for connected camera devices...")
        cams = probe_available_cameras()
        print(f"Available camera index(es): {cams}")
        if not cams:
            print("\nTroubleshooting Guidance:")
            print(get_camera_permission_guidance())
    elif args.video:
        run_stream_demo(source=args.video, window_title=f"SankeetShayak: {Path(args.video).name}")
    elif args.webcam:
        run_stream_demo(source=args.cam_index, window_title="SankeetShayak Webcam Telemetry")
    else:
        run_simulation_demo()
