"""Interactive Demo Runner for HandShayak.

    python demo.py --simulate        physics + classifier demo (no camera, no model download)
    python demo.py --selftest        run the REAL vision pipeline on a scripted hand, write an annotated video
    python demo.py --webcam          live webcam tracking with telemetry HUD
    python demo.py --video clip.mp4  analyse a recorded video
    python demo.py --check-env       verify OpenCV / OpenGL / MediaPipe system libraries
    python demo.py --probe-cameras   list working cameras

Works on macOS, Windows and Linux (with or without a display).
"""

import argparse
from pathlib import Path
import sys

from src.system.render_env import RenderLibraryError, apply_env_defaults

apply_env_defaults()


def run_simulation_demo() -> None:
    from src.data.collector import GestureDatasetBuilder
    from src.models.predictor import GesturePredictor

    print("=" * 65)
    print("HANDSHAYAK: PHYSICS-INFORMED SIGN RECOGNITION DEMO")
    print("=" * 65)
    print("\n1. Initializing gesture predictor & physics engine...")
    predictor = GesturePredictor()

    print("2. Generating test gestures (idle, gentle/forceful tap, slow/rapid swipe)...")
    trajs, labels, _ = GestureDatasetBuilder.generate_synthetic_gestures(
        samples_per_class=1, fps=30.0, frames_per_sample=30, seed=2024)

    correct = 0
    for traj, true_label in zip(trajs, labels):
        pred = predictor.predict_trajectory(traj)
        correct += pred.label == true_label
        print(f"\n> Ground truth : {true_label.upper()}")
        print(f"  Predicted    : {pred.label.upper()} (confidence {pred.confidence * 100:.1f}%)")
        t = pred.telemetry
        print(f"    peak speed {t['peak_speed_m_s']:.3f} m/s | peak accel {t['peak_accel_m_s2']:.1f} m/s^2 | "
              f"peak jerk {t['peak_jerk_m_s3']:.0f} m/s^3")
        print(f"    kinetic energy {t['peak_ke_joules'] * 1000:.1f} mJ | inferred force {t['peak_force_n']:.2f} N")
    print("\n" + "=" * 65)
    print(f"Simulation complete: {correct}/{len(labels)} correct. Dynamics separate gentle from forceful motion.")
    print("=" * 65)


def run_selftest(out_dir: str = "outputs") -> int:
    """Real StreamProcessor + visualiser + video writer driven by a scripted hand (no camera/model needed).

    Script: a gentle tap followed by a forceful tap of the same open hand. The run passes if the measured peak
    hand speed of the second tap is clearly higher than the first.
    """
    import numpy as np
    from src.data.collector import GestureDatasetBuilder
    from src.vision.detector import HandLandmarkDetector, SyntheticBackend
    from src.vision.render import VideoSink, blank_canvas, save_image
    from src.vision.stream_processor import StreamProcessor

    ds = GestureDatasetBuilder.generate_synthetic_dataset(samples_per_class=1, participants=1, seed=7)
    by = {l: t for t, l in zip(ds.trajectories, ds.labels)}
    gentle, forceful = by["tap_gentle"].copy(), by["tap_forceful"].copy()
    forceful += gentle[-1, 0] - forceful[0, 0]            # continuous: no teleport between the two taps
    traj = np.concatenate([gentle, forceful])
    n1 = len(gentle)

    w, h = 640, 480
    detector = HandLandmarkDetector.with_backend(SyntheticBackend(traj, w, h))
    out = Path(out_dir)
    peaks = [0.0, 0.0]
    with StreamProcessor(detector=detector, fps=30.0) as sp, VideoSink(out / "selftest.mp4", 30.0, (w, h)) as sink:
        for i in range(len(traj)):
            frame, tel = sp.process_frame(blank_canvas(w, h), timestamp_sec=i / 30.0)
            sink.write(frame)
            k = 0 if i < n1 else 1
            if i >= 6:                                      # let the 7-frame filter window fill
                peaks[k] = max(peaks[k], tel["speed_m_s"])
        save_image(out / "selftest_last.png", frame)
        print(f"Wrote {sink.path} ({sink.frames_written} frames, codec {sink.codec})")
    print(f"Peak hand speed: gentle tap {peaks[0]:.2f} m/s, forceful tap {peaks[1]:.2f} m/s")
    ok = peaks[1] > 2.0 * peaks[0] > 0.1
    print("SELFTEST " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


def run_stream_demo(source, window_title: str = "HandShayak Live Telemetry") -> int:
    from src.vision.stream_processor import StreamProcessor

    print(f"\nInitializing stream from source: {source!r} ... (press q in the window to quit, Ctrl+C otherwise)")
    try:
        with StreamProcessor(fps=30.0, enable_predictor=True) as processor:
            for frame_idx, _frame, telemetry in processor.process_source(source=source, display=True, window_title=window_title):
                if frame_idx % 30 == 0 and telemetry["speed_m_s"] > 0.05:
                    print(f"[Frame {frame_idx:05d}] speed {telemetry['speed_m_s']:.2f} m/s | "
                          f"KE {telemetry['ke_joules'] * 1000:.1f} mJ | force {telemetry['force_n']:.2f} N")
    except KeyboardInterrupt:
        print("\nStopped.")
    except (RuntimeError, FileNotFoundError) as exc:
        print("\n" + "=" * 65 + f"\nCAMERA / STREAM / MODEL ERROR:\n{exc}\n" + "=" * 65)
        return 1
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="HandShayak Demo Runner")
    parser.add_argument("--simulate", action="store_true", help="physics + classifier demo (headless)")
    parser.add_argument("--selftest", action="store_true", help="run the real vision pipeline on a scripted hand")
    parser.add_argument("--webcam", action="store_true", help="live webcam tracking & physics HUD")
    parser.add_argument("--cam-index", type=int, default=0, help="camera device index (default 0)")
    parser.add_argument("--video", type=str, default=None, help="path to a video file to analyse")
    parser.add_argument("--probe-cameras", action="store_true", help="list working cameras")
    parser.add_argument("--check-env", action="store_true", help="verify OpenCV/OpenGL/MediaPipe system libraries")
    args = parser.parse_args(argv)

    try:
        if args.check_env:
            from scripts.check_env import main as check_main
            return check_main([])
        if args.probe_cameras:
            from src.vision.camera import get_camera_permission_guidance, probe_available_cameras
            cams = probe_available_cameras()
            print(f"Available camera index(es): {cams}")
            if not cams:
                print("\nTroubleshooting guidance:\n" + get_camera_permission_guidance())
            return 0
        if args.video:
            return run_stream_demo(args.video, f"HandShayak: {Path(args.video).name}")
        if args.webcam:
            return run_stream_demo(args.cam_index, "HandShayak Webcam Telemetry")
        if args.selftest:
            return run_selftest()
        run_simulation_demo()
        return 0
    except RenderLibraryError as exc:
        print(f"\nSystem rendering libraries missing:\n{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
