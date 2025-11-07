# NASA Climate Prediction API - Docker Image

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY model_bundle/requirements_deployment.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements_deployment.txt

# Copy model bundle
COPY model_bundle/ /app/model_bundle/

# Copy API code
COPY api/ /app/api/

# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV MODEL_BUNDLE_DIR=/app/model_bundle

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Run the API server
CMD ["python", "api/api_server.py"]
