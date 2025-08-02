
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import subprocess
import tempfile
import os
from tqdm import tqdm
import random

def collate_fn(batch):
    """
    Custom collate function to handle variable-length sequences.
    Pads sequences to the maximum length in the batch.
    """
    clean_embeddings = [item['clean_embedding'] for item in batch]
    noisy_embeddings = [item['noisy_embedding'] for item in batch]
    
    # Find maximum sequence length in this batch
    max_len = max(max(e.shape[0] for e in clean_embeddings),
                  max(e.shape[0] for e in noisy_embeddings))
    
    # Pad sequences to max length
    clean_padded = []
    noisy_padded = []
    masks = []
    
    for clean, noisy in zip(clean_embeddings, noisy_embeddings):
        # Pad clean embedding
        clean_len = clean.shape[0]
        if clean_len < max_len:
            padding = torch.zeros(max_len - clean_len, clean.shape[1])
            clean = torch.cat([clean, padding], dim=0)
        clean_padded.append(clean)
        
        # Pad noisy embedding
        noisy_len = noisy.shape[0]
        if noisy_len < max_len:
            padding = torch.zeros(max_len - noisy_len, noisy.shape[1])
            noisy = torch.cat([noisy, padding], dim=0)
        noisy_padded.append(noisy)
        
        # Create mask (1 for real data, 0 for padding)
        mask = torch.ones(max_len)
        mask[clean_len:] = 0
        masks.append(mask)
    
    return {
        'clean_embedding': torch.stack(clean_padded),
        'noisy_embedding': torch.stack(noisy_padded),
        'mask': torch.stack(masks)
    }


class DenoiseEncoder(nn.Module):
    """
    Denoise Encoder that cleans embeddings from pre-trained audioencoders.
    As per the paper, this can be as simple as MLPs or more complex like ViT layers.
    """
    
    def __init__(self, embedding_dim=1024, hidden_dim=1024, num_layers=3, use_vit=False):
        super().__init__()
        
        if use_vit:
            # Simple ViT-style transformer layers
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                activation='gelu',
                batch_first=True,
                norm_first=True  # Pre-normalization as mentioned in paper
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        else:
            # Simple MLP layers
            layers = []
            for i in range(num_layers):
                if i == 0:
                    layers.append(nn.Linear(embedding_dim, hidden_dim))
                else:
                    layers.append(nn.Linear(hidden_dim, hidden_dim))
                
                if i < num_layers - 1:  # No activation after last layer
                    layers.append(nn.GELU())
                    layers.append(nn.LayerNorm(hidden_dim))
            
            # Final projection back to embedding dimension
            if hidden_dim != embedding_dim:
                layers.append(nn.Linear(hidden_dim, embedding_dim))
            
            self.encoder = nn.Sequential(*layers)
        
        self.use_vit = use_vit
    
    def forward(self, x):
        """
        x: (batch_size, sequence_length, embedding_dim)
        """
        if self.use_vit:
            return self.encoder(x)
        else:
            return self.encoder(x)


class AudioDegradationDataset(Dataset):
    """
    Dataset that creates noisy audio by applying various degradation transforms
    to clean audio, specifically for enhancing TTS outputs.
    """
    
    def __init__(self, audio_dir, knn_vc_model, degradation_types=['mp3', 'mulaw', 'quantize'], 
                 max_samples=None, cache_features=True):
        self.audio_dir = Path(audio_dir)
        self.audio_paths = list(self.audio_dir.glob("*.wav"))
        
        if max_samples:
            self.audio_paths = self.audio_paths[:max_samples]
            
        print(f"Found {len(self.audio_paths)} audio files")
        
        self.knn_vc = knn_vc_model
        self.degradation_types = degradation_types
        self.cache_features = cache_features
        self.feature_cache = {}
        
        # Move model to CPU for multiprocessing compatibility
        self.knn_vc.device = 'cpu'
        self.knn_vc.wavlm = self.knn_vc.wavlm.cpu()
        
        # Pre-cache clean features if requested
        if cache_features:
            print("Pre-caching clean audio features...")
            for path in tqdm(self.audio_paths):
                with torch.no_grad():
                    self.feature_cache[str(path)] = self.knn_vc.get_features(str(path)).cpu()
    
    def degrade_audio(self, wav_path, degradation_type):
        """Apply various audio degradations to simulate TTS artifacts"""
        
        waveform, sr = torchaudio.load(wav_path)
        
        # Ensure 16kHz as required by KNN-VC
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            waveform = resampler(waveform)
            sr = 16000
        
        if degradation_type == 'mp3':
            # Low bitrate MP3 encoding (32 kbps)
            try:
                with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_mp3:
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_wav:
                        torchaudio.save(tmp_wav.name, waveform, sr)
                        # Use ffmpeg for MP3 encoding
                        result = subprocess.run([
                            'ffmpeg', '-i', tmp_wav.name, '-b:a', '32k', 
                            '-y', tmp_mp3.name
                        ], capture_output=True, text=True)
                        
                        if result.returncode != 0:
                            raise Exception(f"ffmpeg error: {result.stderr}")
                            
                        # Convert back to wav
                        subprocess.run([
                            'ffmpeg', '-i', tmp_mp3.name, '-ar', str(sr),
                            '-y', tmp_wav.name
                        ], capture_output=True)
                        degraded, _ = torchaudio.load(tmp_wav.name)
                        os.unlink(tmp_mp3.name)
                        os.unlink(tmp_wav.name)
            except:
                # Fallback to quantization if ffmpeg fails
                degraded = torch.round(waveform * 32) / 32
            
        elif degradation_type == 'mulaw':
            # Severe μ-law encoding (simulate low-quality vocoding)
            # Quantize to 8-bit equivalent
            mulaw_encoded = torchaudio.functional.mu_law_encoding(waveform, 256)
            degraded = torchaudio.functional.mu_law_decoding(mulaw_encoded, 256)
            
        elif degradation_type == 'quantize':
            # Heavy quantization (simulate low-bit vocoder)
            levels = random.choice([16, 32, 64])
            degraded = torch.round(waveform * levels) / levels
            
        elif degradation_type == 'bandlimit':
            # Aggressive low-pass filtering (simulate limited bandwidth vocoder)
            cutoff = random.choice([3000, 4000, 5000])
            degraded = torchaudio.functional.lowpass_biquad(waveform, sr, cutoff)
            
        else:
            # Default: add quantization noise
            degraded = torch.round(waveform * 128) / 128
        
        # Add slight noise to make it more realistic
        noise_level = random.uniform(0.001, 0.005)
        degraded = degraded + torch.randn_like(degraded) * noise_level
        
        # Save degraded audio temporarily
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            torchaudio.save(tmp.name, degraded, sr)
            return tmp.name
    
    def __len__(self):
        return len(self.audio_paths)
    
    def __getitem__(self, idx):
        clean_path = str(self.audio_paths[idx])
        
        # Get clean features from cache or compute
        if self.cache_features and clean_path in self.feature_cache:
            clean_features = self.feature_cache[clean_path]
        else:
            with torch.no_grad():
                clean_features = self.knn_vc.get_features(clean_path).cpu()
        
        # Randomly select degradation type
        degradation_type = np.random.choice(self.degradation_types)
        
        # Create degraded version
        degraded_path = self.degrade_audio(clean_path, degradation_type)
        
        # Extract features from degraded audio
        with torch.no_grad():
            noisy_features = self.knn_vc.get_features(degraded_path).cpu()
        
        # Clean up temporary file
        os.unlink(degraded_path)
        
        return {
            'clean_embedding': clean_features,
            'noisy_embedding': noisy_features
        }


def train_denoise_encoder(audio_dir, num_epochs=50, batch_size=8, device='cuda'):
    """
    Main training function for the denoise encoder.
    """
    # Load pre-trained models
    print("Loading pre-trained KNN-VC model...")
    knn_vc = torch.hub.load('bshall/knn-vc', 'knn_vc', prematched=False, trust_repo=True, pretrained=True)
    
    # Initialize denoise encoder (3-layer MLP as suggested)
    denoise_encoder = DenoiseEncoder(
        embedding_dim=1024,  # KNN-VC WavLM embedding dimension
        hidden_dim=1024,
        num_layers=3,
        use_vit=False  # Use MLP for simplicity
    ).to(device)
    
    print(f"Denoise Encoder Parameters: {sum(p.numel() for p in denoise_encoder.parameters())/1e6:.2f}M")
    
    # Create dataset
    dataset = AudioDegradationDataset(
        audio_dir, 
        knn_vc,
        degradation_types=['mp3', 'mulaw', 'quantize', 'bandlimit'],
        cache_features=True  # Cache clean features for faster training
    )
    
    # Split into train/val
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
    
    # Setup training
    optimizer = optim.AdamW(denoise_encoder.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    mse_loss = nn.MSELoss()
    
    best_val_loss = float('inf')
    
    print("\nStarting training...")
    for epoch in range(num_epochs):
        # Training phase
        denoise_encoder.train()
        train_loss = 0
        train_steps = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in pbar:
            clean_embeddings = batch['clean_embedding'].to(device)
            noisy_embeddings = batch['noisy_embedding'].to(device)
            mask = batch['mask'].to(device)
            
            # Forward pass
            denoised_embeddings = denoise_encoder(noisy_embeddings)
            
            # Compute masked MSE loss (only on non-padded positions)
            loss_per_element = (denoised_embeddings - clean_embeddings) ** 2
            loss_per_element = loss_per_element.mean(dim=-1)  # Average over embedding dimension
            masked_loss = (loss_per_element * mask).sum() / mask.sum()  # Average over valid positions
            
            # Backward pass
            optimizer.zero_grad()
            masked_loss.backward()
            torch.nn.utils.clip_grad_norm_(denoise_encoder.parameters(), 1.0)
            optimizer.step()
            
            train_loss += masked_loss.item()
            train_steps += 1
            
            pbar.set_postfix({'loss': f"{masked_loss.item():.4f}"})
        
        # Validation phase
        denoise_encoder.eval()
        val_loss = 0
        val_steps = 0
        
        with torch.no_grad():
            for batch in val_loader:
                clean_embeddings = batch['clean_embedding'].to(device)
                noisy_embeddings = batch['noisy_embedding'].to(device)
                mask = batch['mask'].to(device)
                
                denoised_embeddings = denoise_encoder(noisy_embeddings)
                
                # Compute masked MSE loss
                loss_per_element = (denoised_embeddings - clean_embeddings) ** 2
                loss_per_element = loss_per_element.mean(dim=-1)
                masked_loss = (loss_per_element * mask).sum() / mask.sum()
                
                val_loss += masked_loss.item()
                val_steps += 1
        
        avg_train_loss = train_loss / train_steps
        avg_val_loss = val_loss / val_steps
        
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': denoise_encoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, 'models/best_denoise_encoder.pth')
            print(f"Saved best model with val loss: {best_val_loss:.4f}")
        
        scheduler.step()
    
    print("\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    return denoise_encoder


if __name__ == "__main__":
    # Bengali TTS dataset path
    audio_dir = "<path to audio directory>"
    
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Train the model
    trained_model = train_denoise_encoder(
        audio_dir=audio_dir,
        num_epochs=50,
        batch_size=8,  # Adjust based on GPU memory
        device=device
    )
    
    print("\nModel training complete! The best model has been saved as 'models/best_denoise_encoder.pth'")