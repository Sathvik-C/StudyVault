# ── Stage 1: Build dependencies ──────────────────
FROM python:3.11-slim AS builder

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# Install system deps needed for building Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: Runtime ─────────────────────────────
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Install runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY . .

# Create directories for ephemeral file storage
RUN mkdir -p /app/uploads /app/storage

# Render uses PORT env var (default 10000)
ENV PORT=10000
EXPOSE ${PORT}

# Start with gunicorn for production
CMD gunicorn main:app \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:${PORT} \
    --workers 1 \
    --timeout 120 \
    --graceful-timeout 30
