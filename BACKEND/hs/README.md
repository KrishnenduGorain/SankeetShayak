# 🖐 HandShayak — physics-informed sign recognition

Hand = rigid-body linkage. MediaPipe landmarks → metres → Savitzky–Golay derivatives (velocity, acceleration, jerk) →
rigid-body dynamics (KE, momentum, force, torque, impulse) → shape-only / dynamics-only / combined classifiers + ablation study.

| Part | Where | Runs on |
|---|---|---|
| Python pipeline, REST + WebSocket API, tests, CI | `src/`, `demo.py`, `tests/` | Linux, macOS, Windows |
| Browser prototype (camera in the browser) | `web/` | any browser → **deploy to Vercel** |

> **Vercel hosts only the browser prototype** (`web/`). Serverless functions have no camera and cannot hold OpenCV + MediaPipe.
> Deploy the Python API with Docker (`Dockerfile`) on Render / Fly / Cloud Run / any VM.

## Quick start (Python)

```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
bash scripts/install_system_deps.sh                       # Linux only: libGL/libEGL/libGLESv2/GLib/X11 (see below)
pip install -r requirements-dev.txt
python scripts/check_env.py --strict                      # verifies OpenCV + OpenGL + MediaPipe
python scripts/download_model.py                          # MediaPipe hand model (also auto-downloaded on first use)
python -m pytest                                          # full test suite
python demo.py --simulate                                 # physics + classifier, no camera
python demo.py --selftest                                 # real vision pipeline on a scripted hand -> outputs/selftest.mp4
python demo.py --webcam                                   # live tracking + telemetry HUD   (q to quit)
python demo.py --video clip.mp4
uvicorn src.api.main:app --reload                         # API on :8000  (docs at /docs)
```

## System-level rendering libraries (the CI gotcha)

`mediapipe` depends on `opencv-contrib-python`; both link **system** libraries that pip does not install:

| Library | apt package | Needed by |
|---|---|---|
| `libGL.so.1` | `libgl1` | OpenCV |
| `libEGL.so.1`, `libGLESv2.so.2` | `libegl1`, `libgles2` | MediaPipe (`libmediapipe.so`) |
| `libglib-2.0.so.0`, `libgthread-2.0.so.0` | `libglib2.0-0t64` (24.04) / `libglib2.0-0` | OpenCV |
| `libX11.so.6`, `libxcb.so.1` | `libx11-6`, `libxcb1` | OpenCV windows |

Bare CI runners and `python:*-slim` images lack them → `ImportError: libGL.so.1: cannot open shared object file`.
macOS and Windows wheels bundle everything.

* `scripts/install_system_deps.sh` installs them (apt / dnf / apk / pacman; `--xvfb` adds a virtual display; `--all` adds optional libs).
* `src/system/render_env.py` checks them **before** `cv2`/`mediapipe` are imported and raises `RenderLibraryError` with the exact install command; it also probes OpenGL/EGL (in a child process) and decides whether a GUI window is safe.
* `src/vision/render.py` — `SafeDisplay` (never aborts without a display), `VideoSink` (mp4v → MJPG → XVID fallback), offscreen skeleton rendering.
* **Headless Linux gotcha:** `cv2.imshow` without an X server does not raise — Qt *aborts the whole process*. The pipeline checks the display first; set `HANDSHAYAK_HEADLESS=1` to force headless (CI does).
* **Do not install `opencv-python` / `opencv-python-headless` next to `mediapipe`** — they overwrite each other's `cv2` files. `requirements.txt` uses the one package MediaPipe already requires.

## CI (`.github/workflows/ci.yml`)
`python` matrix (Ubuntu 22.04/24.04, Windows, macOS × Python 3.11/3.12) · `gui-xvfb` (real OpenCV window on a virtual display) · `missing-libs` (bare container must fail with a helpful message, then the installer must fix it) · `docker` (builds and tests the image) · `web` (Node tests).

## Bugs fixed relative to the first code drop
1. **Physics ran on wrist-centred, palm-length-scaled landmarks**: no global motion and wrong units, so taps measured ≈0 m/s. Now `MetricScaler` converts to metres and keeps global motion (`LandmarkFrame.metric_landmarks`); the old normalised array is for *shape* only.
2. `opencv-python-headless` + `mediapipe` → conflicting `cv2` packages; headless OpenCV also has no `imshow`.
3. `python-dotenv` was imported but not in `requirements.txt` (now listed and optional at runtime).
4. Missing modules created: `src/data/{collector,aggregator}.py`, `src/vision/{metric,render}.py`, `src/system/render_env.py`.
5. `cv2.imshow` on headless Linux aborted the process → `SafeDisplay`.
6. Jerk was a finite difference of a noisy acceleration with polyorder 2 → true 3rd Savitzky–Golay derivative (polyorder 3); irregular webcam timestamps are resampled to a uniform grid.
7. Ablation used a random split → participant-independent `GroupShuffleSplit`, as the report requires.
8. `extract_from_webcam` hard-coded `CAP_AVFOUNDATION` (macOS only) → per-OS backends.
9. CORS: `*` combined with credentials → fixed; WebSocket swallowed all errors → now reports them to the client.
10. Rotational KE, torque and impulse (report eqs. 6–15) were missing → `DynamicsEngine`.

## API
`GET /health` · `GET /signs` · `POST /predict` · `WS /ws/stream`
Landmarks are `(21, 3)` per frame; send `"space": "metric"` (metres) or `"space": "image"` with `width`/`height` (MediaPipe normalised coords; the server converts).

## Web prototype & Vercel
```bash
cd web && npm start        # http://localhost:3000  (camera needs localhost or https; never file://)
cd web && node --test      # 16 physics tests
```
Deploy: import the repo in Vercel (preset **Other**) — `vercel.json` serves `web/`. Or `npx vercel --prod`.

## Honest limits
Force/torque are inferred from motion and assumed anthropometric masses (not measured). Speeds depend on the 9.5 cm palm-length scale
and are best for motion across the image plane. The shipped model is trained on **synthetic** gestures; record real data
(`FeatureAggregator` + `HandLandmarkDetector.extract_from_video(metric=True)`) and retrain with `python -m src.models.train` for real use.
