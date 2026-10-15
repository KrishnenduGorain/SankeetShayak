#!/usr/bin/env python3
"""Verify the OpenCV / OpenGL / MediaPipe system stack.

    python scripts/check_env.py                 # human-readable report
    python scripts/check_env.py --strict        # exit 1 if anything required is missing (use in CI)
    python scripts/check_env.py --json          # machine-readable
    python scripts/check_env.py --fix           # try to install missing system libraries (Linux)
    python scripts/check_env.py --require-display   # also fail without a usable GUI (xvfb jobs)
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.system.render_env import (  # noqa: E402  (does not import cv2)
    environment_report, install_command, install_system_libs,
)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--fix", action="store_true", help="install missing system libraries (Linux; may need sudo)")
    ap.add_argument("--require-display", action="store_true")
    args = ap.parse_args(argv)

    if args.fix and not install_system_libs():
        print("Automatic installation failed; see the command above.", file=sys.stderr)
    rep = environment_report()

    if args.json:
        print(json.dumps(rep.to_dict(), indent=2, default=str))
    else:
        yes = lambda b: "OK     " if b else "MISSING"
        print(f"Platform : {rep.platform} ({rep.machine}), Python {rep.python}")
        if rep.libs.checked:
            print(f"System libs [{yes(rep.libs.ok).strip()}]  present: {', '.join(rep.libs.present) or '-'}")
            if rep.libs.missing_required:
                print(f"  missing REQUIRED : {', '.join(rep.libs.missing_required)}")
                print(f"  fix              : {install_command(rep.libs)}")
            if rep.libs.missing_optional:
                print(f"  missing optional : {', '.join(rep.libs.missing_optional)}")
        else:
            print("System libs [n/a    ]  wheels bundle their libraries on this OS")
        gl = rep.opengl
        print(f"OpenGL   : {gl.api}: libs {'loadable' if gl.libs_loadable else 'NOT loadable'}; {gl.detail}")
        print(f"OpenCV   : {yes(rep.cv2_ok)} {rep.cv2_version} gui={rep.cv2_gui or '-'} {rep.cv2_error}")
        print(f"MediaPipe: {yes(rep.mediapipe_ok)} {rep.mediapipe_version} {rep.mediapipe_error}")
        print(f"Display  : {'headless' if rep.headless else 'available'} - {rep.display_reason}")

    failed = not rep.ok
    if args.require_display and rep.headless:
        print("A usable display is required but none was found.", file=sys.stderr)
        failed = True
    if args.strict and failed:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
