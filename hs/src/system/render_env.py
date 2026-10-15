"""System-level rendering environment: OpenCV / OpenGL / EGL library checks.

WHY THIS EXISTS
---------------
``opencv-contrib-python`` (a hard dependency of MediaPipe) and the MediaPipe native
library link against *system* libraries that are NOT shipped in the Python wheels:

* ``libGL.so.1``                      (OpenGL)       -> cv2
* ``libEGL.so.1``, ``libGLESv2.so.2`` (OpenGL ES/EGL) -> ``mediapipe/tasks/c/libmediapipe.so``
* ``libglib-2.0.so.0``, ``libgthread-2.0.so.0``       -> cv2
* ``libX11.so.6``, ``libxcb.so.1``                    -> cv2 (GUI windows)

On a bare CI runner or a ``python:*-slim`` Docker image these are missing, and
``import cv2`` / ``import mediapipe`` fail with an unhelpful ``ImportError``. Worse,
calling ``cv2.imshow`` on a Linux machine with no display does not raise: the Qt backend
aborts the whole process.  This module detects both problems *before* they bite, using
only ``ctypes`` / the standard library (it never imports ``cv2`` at module level), and gives
actionable messages.

Public API
----------
``ensure_render_env(strict)``  apply safe env defaults + verify libraries (called by ``src.vision``)
``check_system_libs()``        -> :class:`LibCheck`
``probe_opengl()``             -> :class:`GLProbe` (EGL/GL/WGL/CGL probe, isolated in a subprocess)
``can_display()``              -> ``(bool, reason)``; never touches cv2 GUI code
``environment_report()``       -> :class:`RenderReport` (everything above + cv2 / mediapipe import status)
``install_command()``          -> shell command that installs the missing packages
"""

from __future__ import annotations

import ctypes
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import warnings
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

__all__ = [
    "RenderLibraryError",
    "SystemLib",
    "LibCheck",
    "GLProbe",
    "RenderReport",
    "LINUX_LIBS",
    "apply_env_defaults",
    "check_system_libs",
    "probe_opengl",
    "can_display",
    "is_headless",
    "cv2_gui_backend",
    "environment_report",
    "install_command",
    "install_system_libs",
    "ensure_render_env",
]


class RenderLibraryError(ImportError):
    """Raised when required system rendering libraries are missing (subclass of ImportError)."""


# --------------------------------------------------------------------------- library table
@dataclass(frozen=True)
class SystemLib:
    key: str
    sonames: Tuple[str, ...]           # any one of these loading is success
    apt: Tuple[str, ...]               # candidate Debian/Ubuntu packages (first that exists wins)
    dnf: str                           # Fedora / RHEL package
    apk: str                           # Alpine package
    purpose: str
    required: bool                     # required => cv2 / mediapipe cannot be imported without it


LINUX_LIBS: Tuple[SystemLib, ...] = (
    SystemLib("opengl", ("libGL.so.1",), ("libgl1",), "mesa-libGL", "mesa-gl",
              "OpenGL; linked by cv2", True),
    SystemLib("egl", ("libEGL.so.1",), ("libegl1",), "mesa-libEGL", "mesa-egl",
              "EGL; linked by MediaPipe (libmediapipe.so)", True),
    SystemLib("gles2", ("libGLESv2.so.2",), ("libgles2",), "mesa-libGLES", "mesa-gles",
              "OpenGL ES 2; linked by MediaPipe (libmediapipe.so)", True),
    SystemLib("glib", ("libglib-2.0.so.0",), ("libglib2.0-0t64", "libglib2.0-0"), "glib2", "glib",
              "GLib; linked by cv2", True),
    SystemLib("gthread", ("libgthread-2.0.so.0",), ("libglib2.0-0t64", "libglib2.0-0"), "glib2", "glib",
              "GThread (part of GLib); linked by cv2", True),
    SystemLib("x11", ("libX11.so.6",), ("libx11-6",), "libX11", "libx11",
              "X11 client; linked by cv2 (GUI windows)", True),
    SystemLib("xcb", ("libxcb.so.1",), ("libxcb1",), "libxcb", "libxcb",
              "XCB; linked by cv2 (GUI windows)", True),
    # Not required by current wheels, but older OpenCV 4.x wheels and some Qt plugins want them.
    SystemLib("sm", ("libSM.so.6",), ("libsm6",), "libSM", "libsm",
              "X session management (older cv2 wheels)", False),
    SystemLib("ice", ("libICE.so.6",), ("libice6",), "libICE", "libice",
              "X ICE (older cv2 wheels)", False),
    SystemLib("xext", ("libXext.so.6",), ("libxext6",), "libXext", "libxext",
              "X extensions (older cv2 wheels)", False),
    SystemLib("xrender", ("libXrender.so.1",), ("libxrender1",), "libXrender", "libxrender",
              "X render (older cv2 wheels)", False),
    SystemLib("portaudio", ("libportaudio.so.2",), ("libportaudio2",), "portaudio", "portaudio",
              "PortAudio; only for MediaPipe *audio* tasks (sounddevice)", False),
)


# --------------------------------------------------------------------------- env defaults
_DEFAULTS_APPLIED = False


def _truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def apply_env_defaults() -> None:
    """Quieten native logs and avoid known slow / crashy defaults. Idempotent; never overrides the user."""
    global _DEFAULTS_APPLIED
    if _DEFAULTS_APPLIED:
        return
    os.environ.setdefault("GLOG_minloglevel", "2")            # MediaPipe / glog
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
    os.environ.setdefault("MEDIAPIPE_DISABLE_GPU", "1")        # CPU inference: reproducible on CI
    if sys.platform.startswith("win"):
        # Media Foundation hardware transforms make webcam start-up take 30+ s on many laptops.
        os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")
    if is_headless():
        os.environ.setdefault("MPLBACKEND", "Agg")
    # NOTE: do NOT set QT_QPA_PLATFORM=offscreen: the Qt bundled with cv2 wheels has no such plugin
    # and aborts. Headless safety is handled by can_display() instead.
    _DEFAULTS_APPLIED = True


# --------------------------------------------------------------------------- library checks
def _try_load(soname: str) -> bool:
    try:
        ctypes.CDLL(soname)
        return True
    except OSError:
        return False


def _lib_present(lib: SystemLib) -> bool:
    return any(_try_load(s) for s in lib.sonames)


def _os_release() -> Dict[str, str]:
    data: Dict[str, str] = {}
    try:
        with open("/etc/os-release", encoding="utf-8") as fh:
            for line in fh:
                if "=" in line:
                    k, v = line.rstrip().split("=", 1)
                    data[k] = v.strip().strip('"')
    except OSError:
        pass
    return data


def _distro_family() -> str:
    """'debian' | 'fedora' | 'alpine' | 'arch' | 'unknown' (Linux only)."""
    osr = _os_release()
    ids = f"{osr.get('ID', '')} {osr.get('ID_LIKE', '')}".lower()
    if any(x in ids for x in ("debian", "ubuntu", "mint", "pop", "raspbian")):
        return "debian"
    if any(x in ids for x in ("fedora", "rhel", "centos", "rocky", "alma", "amzn")):
        return "fedora"
    if "alpine" in ids:
        return "alpine"
    if "arch" in ids or "manjaro" in ids:
        return "arch"
    return "unknown"


def _apt_package(lib: SystemLib) -> str:
    """Pick the first apt candidate that actually exists on this release (t64 rename in Ubuntu 24.04)."""
    if len(lib.apt) == 1 or not shutil.which("apt-cache"):
        return lib.apt[0]
    for cand in lib.apt:
        try:
            out = subprocess.run(["apt-cache", "show", cand], capture_output=True, text=True, timeout=20)
            if out.returncode == 0 and out.stdout.strip():
                return cand
        except (OSError, subprocess.SubprocessError):
            break
    return lib.apt[0]


@dataclass
class LibCheck:
    platform: str
    checked: bool                                  # False on Windows/macOS (wheels bundle everything)
    present: List[str] = field(default_factory=list)
    missing_required: List[str] = field(default_factory=list)
    missing_optional: List[str] = field(default_factory=list)
    packages: List[str] = field(default_factory=list)          # packages needed for missing *required* libs
    optional_packages: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.missing_required


def check_system_libs(libs: Tuple[SystemLib, ...] = LINUX_LIBS) -> LibCheck:
    """Check the system libraries cv2 / MediaPipe need. Only meaningful on Linux."""
    plat = sys.platform
    if not plat.startswith("linux"):
        return LibCheck(platform=plat, checked=False)
    result = LibCheck(platform=plat, checked=True)
    family = _distro_family()
    for lib in libs:
        if _lib_present(lib):
            result.present.append(lib.key)
            continue
        pkg = {"debian": None, "fedora": lib.dnf, "alpine": lib.apk}.get(family)
        if family == "debian":
            pkg = _apt_package(lib)
        pkg = pkg or lib.apt[0]
        if lib.required:
            result.missing_required.append(lib.key)
            if pkg not in result.packages:
                result.packages.append(pkg)
        else:
            result.missing_optional.append(lib.key)
            if pkg not in result.optional_packages:
                result.optional_packages.append(pkg)
    return result


def install_command(check: Optional[LibCheck] = None, include_optional: bool = False) -> str:
    """Shell command that installs everything that is missing (or the full set if nothing is known)."""
    check = check or check_system_libs()
    if sys.platform == "darwin":
        return "# macOS: nothing to install (the pip wheels bundle their libraries)"
    if sys.platform.startswith("win"):
        return "# Windows: nothing to install (the pip wheels bundle their libraries)"
    pkgs = list(check.packages) + (list(check.optional_packages) if include_optional else [])
    family = _distro_family()
    if not pkgs:
        return "# all required system libraries are already installed"
    joined = " ".join(pkgs)
    sudo = "" if (hasattr(os, "geteuid") and os.geteuid() == 0) else "sudo "
    if family == "debian":
        return f"{sudo}apt-get update && {sudo}apt-get install -y --no-install-recommends {joined}"
    if family == "fedora":
        return f"{sudo}dnf install -y {joined}"
    if family == "alpine":
        return f"{sudo}apk add --no-cache {joined}"
    if family == "arch":
        return f"{sudo}pacman -S --needed --noconfirm mesa glib2 libx11 libxcb"
    return "Install the OpenGL/EGL/GLib/X11 runtime packages for your distribution, or run scripts/install_system_deps.sh"


def install_system_libs(include_optional: bool = False, dry_run: bool = False) -> bool:
    """Install missing system libraries (Linux). Returns True when everything required is present afterwards."""
    check = check_system_libs()
    if not check.checked or check.ok and not (include_optional and check.optional_packages):
        return True
    cmd = install_command(check, include_optional=include_optional)
    print(f"+ {cmd}")
    if dry_run:
        return check.ok
    rc = subprocess.call(cmd, shell=True)
    return rc == 0 and check_system_libs().ok


# --------------------------------------------------------------------------- OpenGL / EGL probe
@dataclass
class GLProbe:
    api: str                         # 'EGL' | 'GL' | 'WGL' | 'CGL' | 'none'
    libs_loadable: bool
    context_ok: bool                 # a display/context could actually be initialised
    detail: str = ""

    @property
    def usable_for_rendering(self) -> bool:
        return self.libs_loadable


_EGL_PROBE_CODE = r"""
import ctypes, json
out = {"libs": False, "ctx": False, "detail": ""}
try:
    egl = ctypes.CDLL("libEGL.so.1")
    out["libs"] = True
    egl.eglGetDisplay.restype = ctypes.c_void_p
    egl.eglGetDisplay.argtypes = [ctypes.c_void_p]
    egl.eglInitialize.restype = ctypes.c_uint
    egl.eglInitialize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
    egl.eglTerminate.argtypes = [ctypes.c_void_p]
    dpy = egl.eglGetDisplay(None)
    if not dpy:
        out["detail"] = "eglGetDisplay returned EGL_NO_DISPLAY (no GPU/driver: software paths only)"
    else:
        maj, mnr = ctypes.c_int(), ctypes.c_int()
        if egl.eglInitialize(dpy, ctypes.byref(maj), ctypes.byref(mnr)):
            out["ctx"] = True
            out["detail"] = "EGL %d.%d initialised" % (maj.value, mnr.value)
            egl.eglTerminate(dpy)
        else:
            out["detail"] = "libEGL loads but eglInitialize failed (no GPU/driver: software paths only)"
except OSError as exc:
    out["detail"] = "libEGL.so.1 not loadable: %s" % exc
print(json.dumps(out))
"""


def probe_opengl(timeout: float = 15.0) -> GLProbe:
    """Probe OpenGL/EGL availability without ever risking the host process.

    Linux: ``libEGL`` is loaded and initialised in a *child* process (driver bugs can segfault).
    Windows: ``opengl32.dll``.  macOS: ``OpenGL.framework``.
    """
    if sys.platform.startswith("linux"):
        gl_ok = _try_load("libGL.so.1")
        try:
            proc = subprocess.run([sys.executable, "-c", _EGL_PROBE_CODE], capture_output=True, text=True, timeout=timeout)
            data = json.loads(proc.stdout.strip().splitlines()[-1]) if proc.stdout.strip() else {}
        except (OSError, subprocess.SubprocessError, ValueError, IndexError) as exc:
            return GLProbe("EGL", gl_ok, False, f"EGL probe failed: {exc}")
        libs = bool(data.get("libs")) and gl_ok
        detail = data.get("detail", "") + ("" if gl_ok else "; libGL.so.1 missing")
        return GLProbe("EGL", libs, bool(data.get("ctx")), detail)
    if sys.platform.startswith("win"):
        try:
            ctypes.WinDLL("opengl32")  # type: ignore[attr-defined]
            return GLProbe("WGL", True, True, "opengl32.dll loaded")
        except (OSError, AttributeError) as exc:
            return GLProbe("WGL", False, False, str(exc))
    if sys.platform == "darwin":
        path = "/System/Library/Frameworks/OpenGL.framework/OpenGL"
        ok = os.path.exists(path) or bool(_try_load(path))
        return GLProbe("CGL", ok, ok, "OpenGL.framework" if ok else "OpenGL.framework not found")
    return GLProbe("none", False, False, f"unsupported platform {sys.platform}")


# --------------------------------------------------------------------------- display / headless
def _x11_reachable() -> Tuple[bool, str]:
    """True if an X server accepts a connection (checked through libX11, never through Qt)."""
    if not os.environ.get("DISPLAY"):
        return False, "DISPLAY is not set"
    try:
        x11 = ctypes.CDLL("libX11.so.6")
    except OSError:
        return False, "libX11.so.6 not installed"
    x11.XOpenDisplay.restype = ctypes.c_void_p
    x11.XOpenDisplay.argtypes = [ctypes.c_char_p]
    x11.XCloseDisplay.argtypes = [ctypes.c_void_p]
    dpy = x11.XOpenDisplay(None)
    if not dpy:
        return False, f"cannot connect to X display {os.environ.get('DISPLAY')!r}"
    x11.XCloseDisplay(dpy)
    return True, "X display reachable"


def is_headless() -> bool:
    """True when no interactive window can (or should) be opened."""
    if _truthy("HANDSHAYAK_HEADLESS"):
        return True
    if _truthy("HANDSHAYAK_FORCE_GUI"):
        return False
    if sys.platform.startswith("linux"):
        return not _x11_reachable()[0]
    return False


def cv2_gui_backend() -> Optional[str]:
    """GUI backend of the installed OpenCV ('QT5', 'GTK', 'COCOA', 'WIN32UI', ... or 'NONE'); None if unknown.

    Imports cv2 lazily; returns None if cv2 cannot be imported.
    """
    try:
        import cv2  # noqa: WPS433 (lazy on purpose)
    except Exception:  # pragma: no cover - cv2 missing / broken libs
        return None
    m = re.search(r"^\s*GUI:\s*(\S+)", cv2.getBuildInformation(), re.MULTILINE)
    return m.group(1).upper() if m else None


def can_display() -> Tuple[bool, str]:
    """Can ``cv2.imshow`` be used safely? Never calls any cv2 GUI function."""
    if _truthy("HANDSHAYAK_HEADLESS"):
        return False, "HANDSHAYAK_HEADLESS is set"
    if sys.platform.startswith("linux"):
        ok, why = _x11_reachable()
        if not ok:
            return False, f"{why} (cv2's Qt backend would abort the process; running headless)"
    backend = cv2_gui_backend()
    if backend == "NONE":
        return False, "installed OpenCV has no GUI support (opencv-python-headless?)"
    return True, f"GUI backend {backend or 'unknown'}"


# --------------------------------------------------------------------------- report
@dataclass
class RenderReport:
    platform: str
    python: str
    machine: str
    libs: LibCheck
    opengl: GLProbe
    headless: bool
    display_reason: str
    cv2_ok: bool = False
    cv2_version: str = ""
    cv2_gui: str = ""
    cv2_error: str = ""
    mediapipe_ok: bool = False
    mediapipe_version: str = ""
    mediapipe_error: str = ""

    @property
    def ok(self) -> bool:
        return self.libs.ok and self.cv2_ok and self.mediapipe_ok

    def to_dict(self) -> dict:
        d = asdict(self)
        d["ok"] = self.ok
        return d


def environment_report(import_checks: bool = True) -> RenderReport:
    """Collect everything about the rendering stack. Import checks run in *this* process."""
    apply_env_defaults()
    libs = check_system_libs()
    can, reason = can_display() if import_checks else (False, "not evaluated")
    rep = RenderReport(
        platform=platform.platform(), python=sys.version.split()[0], machine=platform.machine(),
        libs=libs, opengl=probe_opengl(), headless=not can, display_reason=reason,
    )
    if import_checks:
        try:
            import cv2
            rep.cv2_ok, rep.cv2_version = True, cv2.__version__
            rep.cv2_gui = cv2_gui_backend() or ""
        except Exception as exc:  # ImportError / OSError (missing .so)
            rep.cv2_error = f"{type(exc).__name__}: {exc}"
        try:
            import mediapipe
            from mediapipe.tasks import python as _mpp  # noqa: F401
            from mediapipe.tasks.python import vision as _mpv  # noqa: F401
            rep.mediapipe_ok, rep.mediapipe_version = True, getattr(mediapipe, "__version__", "?")
        except Exception as exc:
            rep.mediapipe_error = f"{type(exc).__name__}: {exc}"
    return rep


# --------------------------------------------------------------------------- entry point used by the pipeline
_WARNED = False


def ensure_render_env(strict: bool = True) -> LibCheck:
    """Prepare the process for OpenCV/MediaPipe. Call BEFORE importing cv2 or mediapipe.

    * applies safe environment defaults;
    * verifies required system libraries (Linux);
    * ``strict=True`` raises :class:`RenderLibraryError` with the exact install command when something
      required is missing; ``strict=False`` only warns.
    Set ``HANDSHAYAK_SKIP_LIB_CHECK=1`` to bypass (e.g. exotic distros with non-standard sonames).
    """
    global _WARNED
    apply_env_defaults()
    if _truthy("HANDSHAYAK_SKIP_LIB_CHECK"):
        return LibCheck(platform=sys.platform, checked=False)
    check = check_system_libs()
    if check.ok:
        return check
    msg = (
        "Missing system rendering libraries required by OpenCV/MediaPipe: "
        + ", ".join(check.missing_required)
        + f"\nFix (Linux):\n  {install_command(check)}\n"
        "or run: bash scripts/install_system_deps.sh\n"
        "(Docker/CI images such as python:*-slim and ubuntu runners do not ship these.)"
    )
    if strict:
        raise RenderLibraryError(msg)
    if not _WARNED:
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
        _WARNED = True
    return check
