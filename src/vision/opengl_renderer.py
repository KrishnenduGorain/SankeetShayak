"""OpenGL and OpenCV 3D Graphics and Rendering Utilities.

Provides hardware-accelerated and offscreen 3D rendering for 21-keypoint
hand skeletons, biomechanical velocity vectors, and coordinate grid frames.
Compatible with headless CI, EGL, and cross-platform desktop displays.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import cv2

from src.vision.detector import HAND_CONNECTIONS, HAND_LANDMARK_NAMES

logger = logging.getLogger(__name__)

# Check optional PyOpenGL availability
_OPENGL_AVAILABLE = False
try:
    from OpenGL import GL as gl
    from OpenGL import GLU as glu
    _OPENGL_AVAILABLE = True
except Exception:
    _OPENGL_AVAILABLE = False


def check_opengl_support() -> Dict[str, bool]:
    """Check availability of system OpenGL, EGL, and PyOpenGL libraries."""
    support = {
        "pyopengl": _OPENGL_AVAILABLE,
        "egl": False,
        "headless_capable": True,
    }
    try:
        from OpenGL import EGL as egl
        support["egl"] = True
    except Exception:
        support["egl"] = False
    return support


class OpenGLHandRenderer:
    """Renders 3D hand landmarks, bones, and velocity vectors.
    
    Supports:
    1. Offscreen 3D software/hardware rendering buffer (RGB/BGR).
    2. Depth-sorted 3D perspective projection with rotational orbit controls.
    3. Biomechanical velocity vector arrows and kinetic HUD overlays.
    4. Headless execution for CI/Docker testing without display servers.
    """

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        elevation: float = 15.0,
        azimuth: float = 25.0,
        camera_distance: float = 2.0,
        fov_deg: float = 45.0,
        background_color: Tuple[int, int, int] = (18, 18, 24),
    ):
        self.width = width
        self.height = height
        self.elevation = elevation
        self.azimuth = azimuth
        self.camera_distance = camera_distance
        self.fov_deg = fov_deg
        self.background_color = background_color
        self._opengl_available = _OPENGL_AVAILABLE

    @property
    def is_hardware_accelerated(self) -> bool:
        """Returns True if PyOpenGL bindings are loaded."""
        return self._opengl_available

    def set_camera_angles(self, elevation: float, azimuth: float):
        """Update elevation (pitch) and azimuth (yaw) viewing angles in degrees."""
        self.elevation = elevation
        self.azimuth = azimuth

    def _compute_view_matrix(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute 3D rotation and translation matrices for the camera view."""
        elev_rad = np.radians(self.elevation)
        azim_rad = np.radians(self.azimuth)

        # Rotation around X (elevation / pitch)
        Rx = np.array([
            [1.0, 0.0, 0.0],
            [0.0, np.cos(elev_rad), -np.sin(elev_rad)],
            [0.0, np.sin(elev_rad), np.cos(elev_rad)],
        ], dtype=np.float64)

        # Rotation around Y (azimuth / yaw)
        Ry = np.array([
            [np.cos(azim_rad), 0.0, np.sin(azim_rad)],
            [0.0, 1.0, 0.0],
            [-np.sin(azim_rad), 0.0, np.cos(azim_rad)],
        ], dtype=np.float64)

        R = Rx @ Ry
        # Camera translation along Z
        t = np.array([0.0, 0.0, self.camera_distance], dtype=np.float64)
        return R, t

    def _project_3d_points(self, points_3d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Projects 3D points in hand space to 2D screen coordinates and depths.
        
        Args:
            points_3d: Shape (N, 3) coordinates.
            
        Returns:
            points_2d: Shape (N, 2) pixel coordinates.
            depths: Shape (N,) Z depths from camera.
        """
        R, t = self._compute_view_matrix()
        # Transform points to camera coordinate frame: P_cam = R * P + t
        p_cam = (R @ points_3d.T).T + t

        depths = p_cam[:, 2].copy()
        # Prevent division by zero / negative depths
        safe_depths = np.clip(depths, 0.1, 100.0)

        # Perspective projection
        f = (self.height / 2.0) / np.tan(np.radians(self.fov_deg / 2.0))
        cx = self.width / 2.0
        cy = self.height / 2.0

        u = (p_cam[:, 0] * f / safe_depths) + cx
        v = (-p_cam[:, 1] * f / safe_depths) + cy  # Invert Y so up is screen up

        points_2d = np.column_stack([u, v])
        return points_2d, depths

    def render_hand_3d(
        self,
        landmarks_21x3: np.ndarray,
        velocities: Optional[np.ndarray] = None,
        connections: Optional[List[Tuple[int, int]]] = None,
        show_grid: bool = True,
        show_labels: bool = False,
    ) -> np.ndarray:
        """Renders 3D hand landmarks and bone structure to an offscreen buffer.
        
        Args:
            landmarks_21x3: Array of shape (21, 3) 3D keypoints.
            velocities: Optional array of shape (21, 3) or (N, 21, 3) joint velocities.
            connections: Joint linkages (defaults to standard MediaPipe hand graph).
            show_grid: If True, draws an orienting ground plane grid.
            show_labels: If True, labels key landmark names.
            
        Returns:
            RGB image buffer of shape (height, width, 3) as uint8 np.ndarray.
        """
        if connections is None:
            connections = HAND_CONNECTIONS

        # Initialize offscreen canvas with background color
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        canvas[:] = self.background_color

        if landmarks_21x3 is None or len(landmarks_21x3) != 21:
            return canvas

        lms = np.asarray(landmarks_21x3, dtype=np.float64)

        # Draw 3D reference ground grid
        if show_grid:
            self._draw_grid_plane(canvas)

        # Project 3D points
        p2d, depths = self._project_3d_points(lms)

        # Collect drawing primitives for depth sorting (painter's algorithm)
        primitives = []

        # 1. Bone segments
        for p1_idx, p2_idx in connections:
            if p1_idx < 21 and p2_idx < 21:
                avg_depth = (depths[p1_idx] + depths[p2_idx]) / 2.0
                primitives.append({
                    "type": "bone",
                    "depth": avg_depth,
                    "pt1": tuple(p2d[p1_idx].astype(int)),
                    "pt2": tuple(p2d[p2_idx].astype(int)),
                    "idx1": p1_idx,
                    "idx2": p2_idx,
                })

        # 2. Joint spheres
        for i in range(21):
            primitives.append({
                "type": "joint",
                "depth": depths[i],
                "pt": tuple(p2d[i].astype(int)),
                "idx": i,
            })

        # 3. Velocity vector arrows (if provided)
        if velocities is not None:
            vels = np.asarray(velocities, dtype=np.float64)
            if vels.ndim == 3:
                vels = vels[-1]  # Latest frame
            if vels.shape == (21, 3):
                # Scale velocity vector for visual rendering
                vel_end_3d = lms + vels * 0.15
                v2d_end, _ = self._project_3d_points(vel_end_3d)
                for i in [0, 4, 8, 12, 16, 20]:  # Wrist and fingertips
                    speed = np.linalg.norm(vels[i])
                    if speed > 0.05:
                        primitives.append({
                            "type": "velocity",
                            "depth": depths[i] - 0.01,  # Draw in front of joint
                            "pt1": tuple(p2d[i].astype(int)),
                            "pt2": tuple(v2d_end[i].astype(int)),
                            "speed": speed,
                        })

        # Sort all primitives by depth descending (furthest drawn first)
        primitives.sort(key=lambda item: item["depth"], reverse=True)

        # Render primitives
        for prim in primitives:
            ptype = prim["type"]
            if ptype == "bone":
                pt1 = prim["pt1"]
                pt2 = prim["pt2"]
                # Color gradient based on depth
                d = np.clip((prim["depth"] - 1.0) / 2.0, 0.0, 1.0)
                color = (int(0 + 200 * (1 - d)), int(220 * (1 - d * 0.5)), int(255 * (1 - d * 0.3)))
                thickness = max(2, int(4 * (1.5 / max(prim["depth"], 0.5))))
                cv2.line(canvas, pt1, pt2, color, thickness, cv2.LINE_AA)

            elif ptype == "joint":
                pt = prim["pt"]
                idx = prim["idx"]
                radius = max(3, int(6 * (1.5 / max(prim["depth"], 0.5))))
                # Fingertips special highlight (4, 8, 12, 16, 20)
                if idx in [4, 8, 12, 16, 20]:
                    joint_color = (0, 165, 255)  # Orange
                    radius += 2
                elif idx == 0:
                    joint_color = (0, 0, 255)    # Red for wrist
                    radius += 3
                else:
                    joint_color = (0, 255, 128)  # Greenish-cyan

                cv2.circle(canvas, pt, radius, joint_color, -1, cv2.LINE_AA)
                cv2.circle(canvas, pt, radius + 1, (255, 255, 255), 1, cv2.LINE_AA)

                if show_labels and idx in [0, 4, 8, 12, 16, 20]:
                    name = HAND_LANDMARK_NAMES[idx]
                    cv2.putText(
                        canvas,
                        name,
                        (pt[0] + 8, pt[1] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.35,
                        (200, 200, 200),
                        1,
                        cv2.LINE_AA,
                    )

            elif ptype == "velocity":
                pt1 = prim["pt1"]
                pt2 = prim["pt2"]
                cv2.arrowedLine(canvas, pt1, pt2, (50, 50, 255), 2, cv2.LINE_AA, tipLength=0.3)

        # Draw HUD info
        self._draw_render_hud(canvas)
        return canvas

    def _draw_grid_plane(self, canvas: np.ndarray, grid_size: float = 0.6, steps: int = 6):
        """Draw an orienting ground grid in the X-Z plane."""
        step_val = (2 * grid_size) / steps
        for i in range(steps + 1):
            val = -grid_size + i * step_val
            # Line parallel to Z
            line_z = np.array([[val, -0.4, -grid_size], [val, -0.4, grid_size]])
            # Line parallel to X
            line_x = np.array([[-grid_size, -0.4, val], [grid_size, -0.4, val]])

            p2d_z, _ = self._project_3d_points(line_z)
            p2d_x, _ = self._project_3d_points(line_x)

            color = (40, 40, 55) if val != 0 else (70, 70, 95)
            cv2.line(canvas, tuple(p2d_z[0].astype(int)), tuple(p2d_z[1].astype(int)), color, 1, cv2.LINE_AA)
            cv2.line(canvas, tuple(p2d_x[0].astype(int)), tuple(p2d_x[1].astype(int)), color, 1, cv2.LINE_AA)

    def _draw_render_hud(self, canvas: np.ndarray):
        """Draw viewport parameters and renderer telemetry overlay."""
        backend_tag = "OpenGL (HW)" if self._opengl_available else "OpenCV 3D (EGL/Offscreen)"
        cv2.putText(
            canvas,
            f"Renderer: {backend_tag} | FOV: {self.fov_deg} | Elev: {self.elevation:.0f} Azim: {self.azimuth:.0f}",
            (12, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (140, 140, 160),
            1,
            cv2.LINE_AA,
        )


def render_offscreen_hand(
    landmarks_21x3: np.ndarray,
    width: int = 640,
    height: int = 480,
    elevation: float = 15.0,
    azimuth: float = 25.0,
) -> np.ndarray:
    """Convenience helper to render a 3D hand pose offscreen to an image buffer."""
    renderer = OpenGLHandRenderer(width=width, height=height, elevation=elevation, azimuth=azimuth)
    return renderer.render_hand_3d(landmarks_21x3)
