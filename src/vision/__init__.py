"""Vision module: 3D Hand Landmark Extraction, Cross-Platform Capture, OpenGL Utilities, and Physics Telemetry."""

from src.vision.camera import (
    CrossPlatformCamera,
    get_camera_permission_guidance,
    get_platform_camera_backends,
    probe_available_cameras,
)
from src.vision.detector import HAND_CONNECTIONS, HAND_LANDMARK_NAMES, HandLandmarkDetector, LandmarkFrame
from src.vision.mediapipeline import MediaPipeline, process_source
from src.vision.opengl_renderer import (
    OpenGLHandRenderer,
    check_opengl_support,
    render_offscreen_hand,
)
from src.vision.stream_processor import StreamProcessor
from src.vision.visualizer import HandVisualizer

__all__ = [
    "HandLandmarkDetector",
    "LandmarkFrame",
    "HAND_CONNECTIONS",
    "HAND_LANDMARK_NAMES",
    "HandVisualizer",
    "CrossPlatformCamera",
    "probe_available_cameras",
    "get_camera_permission_guidance",
    "get_platform_camera_backends",
    "StreamProcessor",
    "OpenGLHandRenderer",
    "check_opengl_support",
    "render_offscreen_hand",
    "MediaPipeline",
    "process_source",
]
