# DermaScan AI - CPU inference image.
#
# The trained checkpoint is NOT baked in: models/latest.pt is 41 MB, gitignored,
# and carries the HAM10000 non-commercial licence. Mount it at run time:
#
#   docker build -t dermascan .
#   docker run -p 8088:8088 \
#       -v "$(pwd)/models:/app/models:ro" \
#       -v dermascan-data:/app/data \
#       dermascan
#
# Without models/ the container still starts and /api/health reports itself
# degraded; the first analysis is what fails, which is the documented behaviour.

FROM python:3.11-slim

# libgomp1 is required by torch; the rest of the default image is enough.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch: the default wheels pull ~2.5 GB of CUDA that never runs here.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        torch torchvision \
    && pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY samples/ ./samples/

# Written to at run time; a named volume should be mounted over it so history
# and uploaded images survive a container replacement.
RUN mkdir -p /app/data/uploads /app/models

# Paths are resolved relative to the application root, never to a host layout.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_PATH=/app/models/latest.pt \
    THRESHOLD_PATH=/app/models/class_thresholds.json \
    CALIBRATION_PATH=/app/models/calibration.json \
    DATABASE_URL=sqlite:////app/data/pathology.db

# Inference is CPU-bound and already bounded internally by
# MAX_CONCURRENT_INFERENCE, so one worker per container is the honest default:
# a second worker would double the resident model without doubling throughput.
# Scale by running more containers.
EXPOSE 8088

RUN useradd --create-home --uid 10001 dermascan \
    && chown -R dermascan:dermascan /app/data /app/models
USER dermascan

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; \
        sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8088/api/health', \
        timeout=4).status == 200 else 1)"

CMD ["python", "-m", "uvicorn", "backend.main:app", \
     "--host", "0.0.0.0", "--port", "8088", "--workers", "1"]
