# Use NVIDIA CUDA base image with Python
FROM nvidia/cuda:11.8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3.10-venv \
    ffmpeg \
    git \
    wget \
    curl \
    build-essential \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python
RUN ln -s /usr/bin/python3.10 /usr/bin/python && \
    ln -s /usr/bin/pip3 /usr/bin/pip

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Upgrade pip and install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Install additional dependencies that might be needed
RUN pip install --no-cache-dir \
    librosa \
    soundfile \
    scipy

# Copy application code
COPY training.py .
COPY gradio-app.py .

# Create necessary directories
RUN mkdir -p /app/models /app/data /app/examples /app/cache

# Set permissions
RUN chmod +x training.py gradio-app.py

# Create a non-root user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Set environment variables for the application
ENV PYTHONPATH=/app
ENV TORCH_HOME=/app/cache
ENV HF_HOME=/app/cache

# Expose port for Gradio
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Default command (can be overridden)
CMD ["python", "gradio-app.py"]