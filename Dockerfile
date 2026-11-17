# Production Dockerfile for Credit Card Fraud Detection Web Service
FROM python:3.10-slim

WORKDIR /app

# Ensure Python output is unbuffered
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy repository source code and assets
COPY . .

# Ensure demo data and trained models are generated if missing
RUN python -c "from src.config import ARTIFACT_PATH; import sys; sys.exit(0 if ARTIFACT_PATH.exists() else 1)" || python train.py --fast

# Expose port
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start the unified FastAPI web service and dashboard (respects cloud PORT)
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}"]
