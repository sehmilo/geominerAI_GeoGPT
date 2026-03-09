FROM python:3.11-slim AS base

# Install system dependencies for GDAL, PostGIS client, and gcc
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libpq-dev libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# Copy application code
COPY core/ /app/core/
COPY backend/ /app/backend/
COPY data/ /app/data/

# Create static directory
RUN mkdir -p /app/backend/static/figures

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
