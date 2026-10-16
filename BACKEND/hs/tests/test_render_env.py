"""System rendering library / OpenGL / display-safety tests (the CI-critical part)."""
import os
import subprocess
import sys
import textwrap

import pytest

from src.system import render_env as re_


def test_lib_table_covers_what_cv2_and_mediapipe_link():
    keys = {l.key for l in re_.LINUX_LIBS if l.required}
    assert {"opengl", "egl", "gles2", "glib", "gthread", "x11", "xcb"} <= keys
    sonames = {s for l in re_.LINUX_LIBS for s in l.sonames}
    assert {"libGL.so.1", "libEGL.so.1", "libGLESv2.so.2", "libglib-2.0.so.0"} <= sonames


def test_check_system_libs_on_this_machine():
    chk = re_.check_system_libs()
    if sys.platform.startswith("linux"):
        assert chk.checked and chk.ok, f"missing: {chk.missing_required}; fix: {re_.install_command(chk)}"
    else:
        assert not chk.checked and chk.ok


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux-only library probing")
def test_missing_libs_are_reported_with_actionable_fix(monkeypatch):
    monkeypatch.setattr(re_, "_try_load", lambda soname: soname not in {"libGL.so.1", "libEGL.so.1"})
    chk = re_.check_system_libs()
    assert not chk.ok and {"opengl", "egl"} <= set(chk.missing_required)
    cmd = re_.install_command(chk)
    assert "libgl1" in cmd and "libegl1" in cmd


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux-only library probing")
def test_strict_raises_and_lenient_warns(monkeypatch):
    monkeypatch.setattr(re_, "_try_load", lambda soname: False)
    monkeypatch.delenv("HANDSHAYAK_SKIP_LIB_CHECK", raising=False)
    with pytest.raises(re_.RenderLibraryError) as ei:
        re_.ensure_render_env(strict=True)
    assert isinstance(ei.value, ImportError) and "install_system_deps.sh" in str(ei.value)
    monkeypatch.setattr(re_, "_WARNED", False)
    with pytest.warns(RuntimeWarning):
        assert not re_.ensure_render_env(strict=False).ok


def test_skip_env_bypasses_check(monkeypatch):
    monkeypatch.setattr(re_, "_try_load", lambda soname: False)
    monkeypatch.setenv("HANDSHAYAK_SKIP_LIB_CHECK", "1")
    assert re_.ensure_render_env(strict=True) is not None


def test_headless_override(monkeypatch):
    monkeypatch.setenv("HANDSHAYAK_HEADLESS", "1")
    ok, why = re_.can_display()
    assert not ok and "HEADLESS" in why and re_.is_headless()


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="DISPLAY semantics are Linux-specific")
def test_no_display_means_no_gui(monkeypatch):
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("HANDSHAYAK_FORCE_GUI", raising=False)
    ok, why = re_.can_display()
    assert not ok and "DISPLAY" in why


def test_opengl_probe_never_raises():
    gl = re_.probe_opengl()
    assert gl.api in {"EGL", "WGL", "CGL", "none"}
    if sys.platform.startswith("linux") and re_.check_system_libs().ok:
        assert gl.libs_loadable, gl.detail


def test_environment_report_serialises():
    rep = re_.environment_report()
    d = rep.to_dict()
    assert d["cv2_ok"] and d["mediapipe_ok"], (rep.cv2_error, rep.mediapipe_error)
    assert isinstance(d["libs"]["present"], list)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux-only")
def test_imshow_without_display_does_not_abort_process():
    """cv2.imshow on headless Linux aborts the interpreter (SIGABRT). SafeDisplay must prevent that."""
    code = textwrap.dedent("""
        import numpy as np
        from src.vision.render import SafeDisplay
        d = SafeDisplay('x')
        assert d.enabled is False
        assert d.show(np.zeros((8, 8, 3), np.uint8)) is False
        assert d.quit_requested() is False
        d.close()
        print('SURVIVED')
    """)
    env = {k: v for k, v in os.environ.items() if k != "DISPLAY"}
    env.pop("WAYLAND_DISPLAY", None)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env,
                         cwd=os.path.dirname(os.path.dirname(__file__)), timeout=60)
    assert out.returncode == 0 and "SURVIVED" in out.stdout, out.stderr


@pytest.mark.gui
def test_safe_display_with_real_display():
    import numpy as np
    from src.vision.render import SafeDisplay
    ok, why = re_.can_display()
    if not ok:
        pytest.skip(why)
    d = SafeDisplay("test")
    assert d.enabled and d.show(np.zeros((32, 32, 3), np.uint8))
    d.close()
