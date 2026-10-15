#!/usr/bin/env bash
# Install the system-level rendering libraries OpenCV and MediaPipe need (libGL, libEGL, libGLESv2, GLib, X11).
#   bash scripts/install_system_deps.sh            install what is needed
#   bash scripts/install_system_deps.sh --xvfb     also install Xvfb (virtual display, for GUI tests in CI)
#   bash scripts/install_system_deps.sh --all      also install optional libs (PortAudio, X session libs)
# macOS and Windows: nothing to do (pip wheels bundle everything).
set -euo pipefail

WITH_XVFB=0; WITH_ALL=0
for a in "$@"; do
  case "$a" in
    --xvfb) WITH_XVFB=1 ;; --all) WITH_ALL=1 ;;
    -h|--help) sed -n '2,7p' "$0"; exit 0 ;;
    *) echo "unknown option: $a" >&2; exit 2 ;;
  esac
done

case "$(uname -s)" in
  Darwin) echo "macOS: no system libraries needed."; exit 0 ;;
  MINGW*|MSYS*|CYGWIN*) echo "Windows: no system libraries needed."; exit 0 ;;
  Linux) ;;
  *) echo "Unsupported OS: $(uname -s)" >&2; exit 1 ;;
esac

SUDO=""
if [ "$(id -u)" -ne 0 ]; then
  if command -v sudo >/dev/null 2>&1; then SUDO="sudo"; else echo "Run as root or install sudo." >&2; exit 1; fi
fi

. /etc/os-release 2>/dev/null || true
FAMILY="${ID_LIKE:-} ${ID:-}"

if command -v apt-get >/dev/null 2>&1; then
  export DEBIAN_FRONTEND=noninteractive
  $SUDO apt-get update -y
  # Ubuntu 24.04 renamed libglib2.0-0 -> libglib2.0-0t64; pick whichever exists on this release.
  if apt-cache show libglib2.0-0t64 >/dev/null 2>&1; then GLIB=libglib2.0-0t64; else GLIB=libglib2.0-0; fi
  PKGS="libgl1 libegl1 libgles2 $GLIB libx11-6 libxcb1"
  [ "$WITH_ALL" = 1 ] && PKGS="$PKGS libsm6 libice6 libxext6 libxrender1 libportaudio2"
  [ "$WITH_XVFB" = 1 ] && PKGS="$PKGS xvfb"
  echo "+ apt-get install $PKGS"
  # shellcheck disable=SC2086
  $SUDO apt-get install -y --no-install-recommends $PKGS
elif command -v dnf >/dev/null 2>&1 || command -v yum >/dev/null 2>&1; then
  PM=$(command -v dnf || command -v yum)
  PKGS="mesa-libGL mesa-libEGL mesa-libGLES glib2 libX11 libxcb"
  [ "$WITH_ALL" = 1 ] && PKGS="$PKGS libSM libICE libXext libXrender portaudio"
  [ "$WITH_XVFB" = 1 ] && PKGS="$PKGS xorg-x11-server-Xvfb"
  echo "+ $PM install $PKGS"
  # shellcheck disable=SC2086
  $SUDO "$PM" install -y $PKGS
elif command -v apk >/dev/null 2>&1; then
  PKGS="mesa-gl mesa-egl mesa-gles glib libx11 libxcb"
  [ "$WITH_XVFB" = 1 ] && PKGS="$PKGS xvfb"
  echo "+ apk add $PKGS (note: Alpine/musl cannot run the manylinux MediaPipe wheels; use a glibc image)"
  # shellcheck disable=SC2086
  $SUDO apk add --no-cache $PKGS
elif command -v pacman >/dev/null 2>&1; then
  PKGS="mesa glib2 libx11 libxcb"
  [ "$WITH_XVFB" = 1 ] && PKGS="$PKGS xorg-server-xvfb"
  echo "+ pacman -S $PKGS"
  # shellcheck disable=SC2086
  $SUDO pacman -S --needed --noconfirm $PKGS
else
  echo "No supported package manager found (apt, dnf, yum, apk, pacman). Install OpenGL/EGL/GLES2/GLib/X11 manually." >&2
  exit 1
fi
echo "System rendering libraries installed."
