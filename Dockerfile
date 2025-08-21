# Multi-stage Dockerfile for development and production
FROM python:3.11-slim AS base

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Development stage
FROM base AS development

# Copy requirements
COPY requirements-dev.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements-dev.txt

# Copy application code
COPY . .

# Expose Reflex port
EXPOSE 3000

# Run Reflex in development mode
CMD ["reflex", "run", "--host", "0.0.0.0"]

# Production stage
FROM base AS production

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY config/ ./config/

# Expose Reflex port
EXPOSE 3000

# Run Reflex in production mode
CMD ["reflex", "run", "--host", "0.0.0.0", "--prod"]