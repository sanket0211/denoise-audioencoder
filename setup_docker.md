# 🐳 Docker Setup Instructions for Audio Denoising Project

This guide provides detailed instructions for building and running the audio denoising project using Docker.

## 📋 Prerequisites

### System Requirements
- Docker Engine 20.10+ 
- Docker Compose 2.0+ (optional)
- NVIDIA Docker support (for GPU acceleration)
- At least 8GB RAM
- 10GB+ free disk space

### GPU Support Setup

#### Install NVIDIA Container Toolkit
```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
      && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
      && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

#### Verify GPU Support
```bash
# Test NVIDIA Docker
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu22.04 nvidia-smi
```

## 🏗️ Building the Docker Image

### Basic Build
```bash
# Clone/navigate to project directory
cd audio-denoising-project

# Build the Docker image
docker build -t audio-denoiser:latest .
```

### Build with Custom Arguments
```bash
# Build with specific CUDA version
docker build --build-arg CUDA_VERSION=11.8 -t audio-denoiser:cuda118 .

# Build with different base image
docker build --build-arg BASE_IMAGE=nvidia/cuda:12.1-devel-ubuntu22.04 -t audio-denoiser:cuda121 .

# Build without cache (clean build)
docker build --no-cache -t audio-denoiser:latest .
```

### Multi-stage Build (Advanced)
```bash
# Build optimized production image
docker build -f Dockerfile.prod -t audio-denoiser:prod .
```

## 🚀 Running Docker Containers

### 1. Training Mode

#### Basic Training
```bash
# Prepare your data directory
mkdir -p $(pwd)/data/audio_files
mkdir -p $(pwd)/models

# Copy your .wav files to data/audio_files/
cp /path/to/your/audio/*.wav $(pwd)/data/audio_files/

# Run training
docker run --gpus all \
  --name audio-training \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -e CUDA_VISIBLE_DEVICES=0 \
  audio-denoiser:latest python training.py
```

#### Training with Custom Configuration
```bash
# Create custom training script
cat > custom_training.py << 'EOF'
import sys
sys.path.append('/app')
from training import train_denoise_encoder

# Custom parameters
trained_model = train_denoise_encoder(
    audio_dir="/app/data/audio_files",
    num_epochs=100,
    batch_size=4,
    device='cuda'
)
EOF

# Run with custom script
docker run --gpus all \
  --name audio-training-custom \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/custom_training.py:/app/custom_training.py \
  audio-denoiser:latest python custom_training.py
```

#### Interactive Training (with monitoring)
```bash
# Run training with interactive monitoring
docker run --gpus all -it \
  --name audio-training-interactive \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/logs:/app/logs \
  audio-denoiser:latest bash

# Inside container:
# python training.py
# tail -f /app/logs/training.log
```

### 2. Demo Application Mode

#### Basic Demo
```bash
# Ensure you have a trained model
ls $(pwd)/models/best_denoise_encoder-prod.pth

# Run Gradio demo
docker run --gpus all \
  --name audio-demo \
  -p 7860:7860 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/examples:/app/examples \
  audio-denoiser:latest python gradio-app.py

# Access demo at http://localhost:7860
```

#### Demo with Custom Port
```bash
# Run on port 8080
docker run --gpus all \
  --name audio-demo-8080 \
  -p 8080:7860 \
  -v $(pwd)/models:/app/models \
  audio-denoiser:latest python gradio-app.py

# Access at http://localhost:8080
```

#### Demo with External Access
```bash
# Allow external connections (be careful with security)
docker run --gpus all \
  --name audio-demo-public \
  -p 0.0.0.0:7860:7860 \
  -v $(pwd)/models:/app/models \
  audio-denoiser:latest python gradio-app.py
```

### 3. Development Mode

#### Interactive Development Container
```bash
# Run container with full project mounted
docker run --gpus all -it \
  --name audio-dev \
  -p 7860:7860 \
  -v $(pwd):/app \
  -w /app \
  audio-denoiser:latest bash

# Inside container, you can:
# - Edit files
# - Run training: python training.py
# - Run demo: python gradio-app.py
# - Debug: python -c "import torch; print(torch.cuda.is_available())"
```

#### Jupyter Notebook Development
```bash
# Install Jupyter in container and run
docker run --gpus all \
  --name audio-jupyter \
  -p 8888:8888 \
  -p 7860:7860 \
  -v $(pwd):/app \
  -w /app \
  audio-denoiser:latest bash -c \
  "pip install jupyter && jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root"

# Access Jupyter at http://localhost:8888
```

## 🔧 Docker Compose (Recommended)

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  audio-training:
    build: .
    image: audio-denoiser:latest
    container_name: audio-training
    command: python training.py
    environment:
      - CUDA_VISIBLE_DEVICES=0
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./logs:/app/logs
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  audio-demo:
    build: .
    image: audio-denoiser:latest
    container_name: audio-demo
    command: python gradio-app.py
    ports:
      - "7860:7860"
    volumes:
      - ./models:/app/models
      - ./examples:/app/examples
    depends_on:
      - audio-training
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  audio-dev:
    build: .
    image: audio-denoiser:latest
    container_name: audio-dev
    command: bash
    stdin_open: true
    tty: true
    ports:
      - "7860:7860"
      - "8888:8888"
    volumes:
      - .:/app
    working_dir: /app
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### Using Docker Compose
```bash
# Build and start all services
docker-compose up --build

# Run specific service
docker-compose up audio-training
docker-compose up audio-demo

# Run in background
docker-compose up -d audio-demo

# View logs
docker-compose logs -f audio-training

# Stop all services
docker-compose down

# Remove volumes (careful!)
docker-compose down -v
```

## 📊 Monitoring and Logs

### View Container Logs
```bash
# Real-time logs
docker logs -f audio-training

# Last 100 lines
docker logs --tail 100 audio-training

# Logs with timestamps
docker logs -t audio-training
```

### Monitor Resource Usage
```bash
# Monitor GPU usage
docker exec audio-training nvidia-smi

# Monitor container stats
docker stats audio-training

# Monitor all containers
docker stats
```

### Access Container Shell
```bash
# Get shell access to running container
docker exec -it audio-training bash

# Run specific command
docker exec audio-training ls -la /app/models
```

## 🔍 Troubleshooting

### Common Issues and Solutions

#### 1. GPU Not Available
```bash
# Check NVIDIA Docker setup
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu22.04 nvidia-smi

# If fails, reinstall NVIDIA Container Toolkit
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

#### 2. Permission Denied Errors
```bash
# Fix volume permissions
sudo chown -R $(id -u):$(id -g) $(pwd)/data $(pwd)/models

# Or run container as current user
docker run --user $(id -u):$(id -g) ... audio-denoiser:latest
```

#### 3. Out of Memory Errors
```bash
# Reduce batch size by editing training.py or using custom script
echo 'batch_size = 2  # Reduced from 8' >> custom_config.py

# Or limit container memory
docker run --memory=8g --gpus all ... audio-denoiser:latest
```

#### 4. Port Already in Use
```bash
# Find process using port
sudo lsof -i :7860

# Kill process or use different port
docker run -p 8080:7860 ... audio-denoiser:latest
```

#### 5. Model File Not Found
```bash
# Check if model exists
ls -la $(pwd)/models/

# Copy model if needed
docker cp audio-training:/app/best_denoise_encoder-prod.pth $(pwd)/models/
```

### Debug Mode
```bash
# Run container in debug mode
docker run --gpus all -it \
  --name audio-debug \
  -v $(pwd):/app \
  --entrypoint bash \
  audio-denoiser:latest

# Inside container, debug step by step:
# python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
# python -c "import torchaudio; print('Audio OK')"
# python -c "from training import DenoiseEncoder; print('Model OK')"
```

## 🧹 Cleanup

### Remove Containers
```bash
# Stop and remove specific container
docker stop audio-training
docker rm audio-training

# Remove all project containers
docker ps -a | grep audio- | awk '{print $1}' | xargs docker rm -f
```

### Remove Images
```bash
# Remove specific image
docker rmi audio-denoiser:latest

# Remove all unused images
docker image prune -a
```

### Complete Cleanup
```bash
# WARNING: This removes ALL unused Docker resources
docker system prune -a --volumes

# Or clean up only this project
docker-compose down -v
docker rmi audio-denoiser:latest
```

## 📈 Performance Optimization

### For Training
```bash
# Use specific GPU
docker run --gpus '"device=1"' ... audio-denoiser:latest

# Increase shared memory
docker run --shm-size=2g ... audio-denoiser:latest

# Use multiple GPUs
docker run --gpus all ... audio-denoiser:latest
```

### For Production
```bash
# Build optimized image
docker build --target production -t audio-denoiser:prod .

# Run with resource limits
docker run --memory=4g --cpus=2 ... audio-denoiser:prod
```

---

**Note**: Always ensure your system meets the GPU requirements and has sufficient resources before running training tasks.