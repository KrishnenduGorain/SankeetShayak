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

# Copy application code and model assets
COPY src/ ./src/
COPY models/ ./models/
COPY demo.py .

# Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose FastAPI port
EXPOSE 8000

# Default command: start FastAPI backend with WebSocket streaming
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
