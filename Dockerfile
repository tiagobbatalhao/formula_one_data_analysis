FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/

# Create cache directory inside container
RUN mkdir -p /app/cache

# Data directory will be mounted as a volume
VOLUME /app/data

# Set Python path to include src directory
ENV PYTHONPATH=/app/src:$PYTHONPATH

# Default entrypoint
ENTRYPOINT ["python"]
