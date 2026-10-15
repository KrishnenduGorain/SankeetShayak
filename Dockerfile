# Multi-stage production Dockerfile for SankeetShayak
FROM python:3.11-slim AS base

# Set working directory & environment variables
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Install system dependencies for OpenCV, OpenGL/EGL, and MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libgl1-mesa-dev \
    libegl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency specifications
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code, static web assets, and model assets
COPY src/ ./src/
COPY models/ ./models/
COPY web/ ./web/
COPY demo.py .

# Expose default port (Render overrides with $PORT dynamically)
EXPOSE 8000

# Default command: support Render $PORT or fallback to 8000
CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
