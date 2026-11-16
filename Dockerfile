FROM python:3.11-slim

WORKDIR /app

# Install system deps needed by lightgbm, pyarrow, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies pinned in requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Source
COPY src/ ./src/
COPY tests/ ./tests/
COPY configs/ ./configs/

# Env
ENV PYTHONPATH=/app
ENV PREFECT_API_URL=http://prefect:4200/api

# Default: run the pipeline once
CMD ["python", "-m", "src.pipeline.flow"]
