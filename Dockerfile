# ==============================================================================
# RetinaFlow Production Dockerfile
# Optimized for AWS (ECS / App Runner) & Azure (Container Apps / App Service)
# ==============================================================================
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    GRADIO_SERVER_NAME="0.0.0.0" \
    GRADIO_SERVER_PORT=7860

# Install system dependencies for image processing and ONNX
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies (cached layer)
COPY requirements-docker.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-docker.txt

# Copy application code, trained ONNX models, and sample assets
COPY app.py .
COPY models/ ./models/
COPY assets/ ./assets/

# Security best practice: Create non-root user
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose the Gradio service port
EXPOSE 7860

# Container healthcheck for AWS/Azure orchestration
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# Launch RetinaFlow
CMD ["python", "app.py"]
