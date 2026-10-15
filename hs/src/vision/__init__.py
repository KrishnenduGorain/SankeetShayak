"""Vision module: 3D hand landmark extraction, cross-platform capture, rendering and physics telemetry.

IMPORTANT: ``src.system.render_env`` is imported (and run) first. It verifies the system rendering libraries
(libGL / libEGL / libGLESv2 / GLib / X11) and sets safe environment defaults *before* OpenCV or
MediaPipe are imported, so a missing library produces an actionable error instead of a cryptic one.
"""

from src.system.render_env import (  # noqa: I001  (order matters)
    RenderLibraryError, can_display, check_system_libs, ensure_render_env, environment_report,
    install_command, probe_opengl,
)

ensure_render_env(strict=True)

from src.vision.camera import (  # noqa: E402
    CrossPlatformCamera, get_camera_permission_guidance, get_platform_camera_backends, probe_available_cameras,
)
from src.vision.detector import (  # noqa: E402
    HAND_CONNECTIONS, HAND_LANDMARK_NAMES, HandLandmarkDetector, LandmarkFrame, ModelUnavailableError,
    SyntheticBackend, ensure_model,
)
from src.vision.metric import MetricScaler  # noqa: E402
from src.vision.render import SafeDisplay, VideoSink, render_skeleton_canvas, save_image  # noqa: E402
from src.vision.stream_processor import StreamProcessor  # noqa: E402
from src.vision.visualizer import HandVisualizer  # noqa: E402

__all__ = [
    "HandLandmarkDetector", "LandmarkFrame", "SyntheticBackend", "ModelUnavailableError", "ensure_model",
    "HAND_CONNECTIONS", "HAND_LANDMARK_NAMES", "HandVisualizer", "MetricScaler",
    "CrossPlatformCamera", "probe_available_cameras", "get_camera_permission_guidance", "get_platform_camera_backends",
    "StreamProcessor", "SafeDisplay", "VideoSink", "render_skeleton_canvas", "save_image",
    "RenderLibraryError", "can_display", "check_system_libs", "ensure_render_env", "environment_report",
    "install_command", "probe_opengl",
]
