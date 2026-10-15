# syntax=docker/dockerfile:1
# glibc-based image: the MediaPipe / OpenCV wheels are manylinux and do not run on Alpine (musl).
FROM python:3.12-slim-bookworm AS base

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HANDSHAYAK_HEADLESS=1

# System-level rendering libraries missing from slim images (libGL, EGL, GLES2, GLib, X11/XCB).
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libegl1 libgles2 libglib2.0-0 libx11-6 libxcb1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src ./src
COPY scripts ./scripts
COPY demo.py pytest.ini ./
RUN mkdir -p models outputs \
 && python scripts/check_env.py --strict \
 && (python scripts/download_model.py || echo "WARNING: hand model not downloaded; it will be fetched on first use")

# ---- test image: docker build --target test -t handshayak-test . && docker run --rm handshayak-test
FROM base AS test
COPY requirements-dev.txt .
RUN pip install -r requirements-dev.txt
COPY tests ./tests
CMD ["python", "-m", "pytest"]

# ---- runtime image (default)
FROM base AS runtime
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
