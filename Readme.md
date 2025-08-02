# 🎙️ TTS Audio Enhancement with Denoise Encoder

A deep learning system for enhancing Text-to-Speech (TTS) audio quality using a lightweight denoise encoder that operates on WavLM embeddings.

## 📋 Overview

This project implements a novel approach to improve TTS audio quality by:
- Training a denoise encoder on WavLM embeddings rather than raw audio
- Using KNN-VC for high-quality vocoding
- Applying various audio degradations during training to simulate real-world TTS artifacts
- Providing an interactive Gradio demo for testing

### Key Features
- **Lightweight Model**: Only ~3M parameters for the denoise encoder
- **High Quality**: Uses WavLM embeddings and HiFi-GAN vocoder
- **Real-time**: Fast inference suitable for production use
- **Flexible**: Supports multiple degradation types (MP3, μ-law, quantization, band-limiting)
- **Interactive Demo**: Easy-to-use web interface

## 🏗️ Architecture

```
Input Audio → WavLM Features → Denoise Encoder → Enhanced Features → KNN-VC Vocoder → Enhanced Audio
```

The system consists of:
1. **WavLM Feature Extractor**: Converts audio to semantic embeddings
2. **Denoise Encoder**: 3-layer MLP that cleans noisy embeddings
3. **KNN-VC Vocoder**: High-quality neural vocoder for audio synthesis

## 📁 Project Structure

```
audio-denoising/
├── training.py              # Main training script
├── gradio-app.py           # Interactive demo application
├── requirements.txt        # Python dependencies
├── Dockerfile             # Docker container setup
├── README.md              # This file
├── models/                # Saved model checkpoints
│   └── best_denoise_encoder-prod.pth
├── data/                  # Training data directory
│   └── audio_files/       # Place your .wav files here
└── examples/              # Example audio files for demo
    ├── example_tts.wav
    └── example_speech.wav
```

## 🚀 Quick Start

### Option 1: Local Installation

#### Prerequisites
- Python 3.10+
- CUDA-capable GPU (recommended)
- FFmpeg installed on your system

#### Install FFmpeg
```bash
# Ubuntu/Debian
sudo apt update && sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Windows - Download from https://ffmpeg.org/download.html
```

#### Setup Environment
```bash
# Create conda environment
conda create --prefix ./venv python=3.10
conda activate ./venv

# Install dependencies
pip install -r requirements.txt
```

#### Train the Model
```bash
# Set your audio directory path in training.py
# Then run training
export CUDA_VISIBLE_DEVICES=0
python training.py
```

#### Run the Demo
```bash
python gradio-app.py
```

### Option 2: Docker Installation

#### Build Docker Image
```bash
docker build -t audio-denoiser .
```

#### Run Training Container
```bash
docker run --gpus all -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models audio-denoiser python training.py
```

#### Run Demo Container
```bash
docker run --gpus all -p 7860:7860 -v $(pwd)/models:/app/models audio-denoiser python gradio-app.py
```

## 🔧 Configuration

### Training Parameters

Edit `training.py` to customize:

```python
# Dataset configuration
audio_dir = "/path/to/your/audio/files"  # Directory with .wav files
max_samples = None                       # Limit dataset size (None = use all)

# Training configuration
num_epochs = 50          # Number of training epochs
batch_size = 8           # Batch size (adjust based on GPU memory)
learning_rate = 1e-4     # Learning rate
device = 'cuda'          # Device to use ('cuda' or 'cpu')

# Model configuration
embedding_dim = 1024     # WavLM embedding dimension
hidden_dim = 1024        # Hidden layer dimension
num_layers = 3           # Number of MLP layers
use_vit = False         # Use Transformer instead of MLP
```

### Degradation Types

The system supports multiple degradation types for training:
- **mp3**: Low bitrate MP3 compression (32 kbps)
- **mulaw**: μ-law encoding (telephone quality)
- **quantize**: Heavy quantization (16/32/64 levels)
- **bandlimit**: Low-pass filtering (3-5 kHz cutoff)

## 📊 Training Process

### Data Preparation
1. Place your clean audio files (.wav format) in a directory
2. Files should ideally be 16kHz mono (will be converted automatically)
3. Recommended: 1000+ audio files for good performance

### Training Pipeline
1. **Feature Extraction**: Convert audio to WavLM embeddings
2. **Degradation**: Apply random degradations to create noisy versions
3. **Training**: Train denoise encoder with MSE loss on embedding pairs
4. **Validation**: Monitor performance on held-out data
5. **Checkpointing**: Save best model based on validation loss

### Monitoring Training
- Training progress is displayed with tqdm progress bars
- Loss values are printed for each epoch
- Best model is automatically saved as `best_denoise_encoder-prod.pth`

## 🎯 Usage Examples

### Command Line Training
```bash
# Basic training
python training.py

# Training with specific GPU
export CUDA_VISIBLE_DEVICES=1
python training.py

# Training with custom parameters (edit training.py)
python training.py
```

### Gradio Demo
```bash
# Start demo server
python gradio-app.py

# Access at http://localhost:7860
```

### Programmatic Usage
```python
from gradio-app import TTSEnhancer

# Initialize enhancer
enhancer = TTSEnhancer(model_path='best_denoise_encoder-prod.pth')

# Enhance audio
original, enhanced, sr = enhancer.enhance_audio('input.wav')

# Apply degradation for testing
degraded_path = enhancer.apply_degradation('input.wav', 'mp3')
```

## 🐳 Docker Usage

### Building the Image
```bash
# Build with default settings
docker build -t audio-denoiser .

# Build with specific Python version
docker build --build-arg PYTHON_VERSION=3.10 -t audio-denoiser .
```

### Running Containers

#### Training
```bash
# Basic training
docker run --gpus all \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  audio-denoiser python training.py

# With custom GPU selection
docker run --gpus '"device=1"' \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -e CUDA_VISIBLE_DEVICES=1 \
  audio-denoiser python training.py
```

#### Demo Application
```bash
# Run demo on port 7860
docker run --gpus all \
  -p 7860:7860 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/examples:/app/examples \
  audio-denoiser python gradio-app.py

# Run on custom port
docker run --gpus all \
  -p 8080:7860 \
  -v $(pwd)/models:/app/models \
  audio-denoiser python gradio-app.py
```

#### Interactive Development
```bash
# Run container with bash shell
docker run --gpus all -it \
  -v $(pwd):/app \
  audio-denoiser bash
```

## 🔍 Troubleshooting

### Common Issues

#### CUDA Out of Memory
```bash
# Reduce batch size in training.py
batch_size = 4  # or 2 for very limited GPU memory
```

#### FFmpeg Not Found
```bash
# Install FFmpeg in container
apt-get update && apt-get install -y ffmpeg

# Or use fallback quantization (automatic in code)
```

#### Model Loading Errors
```bash
# Ensure model file exists and path is correct
ls -la best_denoise_encoder-prod.pth

# Check model path in gradio-app.py
model_path = 'best_denoise_encoder-prod.pth'
```

#### Port Already in Use
```bash
# Use different port
docker run -p 8080:7860 ... 
# Access at http://localhost:8080
```

### Performance Optimization

#### Training Speed
- Use larger batch sizes if GPU memory allows
- Enable feature caching: `cache_features=True`
- Use fewer degradation types for faster data loading

#### Inference Speed
- Use CPU for feature extraction if GPU memory is limited
- Reduce `topk` parameter in KNN matching
- Pre-load models to avoid initialization overhead

## 📈 Monitoring and Evaluation

### Training Metrics
- **Train Loss**: MSE loss on training set
- **Validation Loss**: MSE loss on held-out validation set
- **Model Size**: ~3M parameters for default configuration

### Quality Evaluation
- **Subjective**: Listen to enhanced audio samples
- **Objective**: Compare spectrograms and audio metrics
- **A/B Testing**: Use demo to compare original vs enhanced

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Make your changes
4. Add tests if applicable
5. Commit your changes (`git commit -am 'Add improvement'`)
6. Push to the branch (`git push origin feature/improvement`)
7. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [KNN-VC](https://github.com/bshall/knn-vc) for the high-quality vocoder
- [WavLM](https://github.com/microsoft/unilm/tree/master/wavlm) for robust audio features
- [Gradio](https://gradio.app/) for the interactive demo interface
- PyTorch community for the deep learning framework

## 📧 Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Check the troubleshooting section above
- Review the code comments for implementation details

---

**Note**: This project requires significant computational resources for training. Consider using cloud GPU services if local resources are limited.