import gradio as gr
import torch
import torchaudio
import numpy as np
import tempfile
import os
from pathlib import Path
import subprocess

# Import the model classes from the training script
import torch.nn as nn


class DenoiseEncoder(nn.Module):
    """
    Denoise Encoder that cleans embeddings from pre-trained audioencoders.
    """
    
    def __init__(self, embedding_dim=1024, hidden_dim=1024, num_layers=3, use_vit=False):
        super().__init__()
        
        if use_vit:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                activation='gelu',
                batch_first=True,
                norm_first=True
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        else:
            layers = []
            for i in range(num_layers):
                if i == 0:
                    layers.append(nn.Linear(embedding_dim, hidden_dim))
                else:
                    layers.append(nn.Linear(hidden_dim, hidden_dim))
                
                if i < num_layers - 1:
                    layers.append(nn.GELU())
                    layers.append(nn.LayerNorm(hidden_dim))
            
            if hidden_dim != embedding_dim:
                layers.append(nn.Linear(hidden_dim, embedding_dim))
            
            self.encoder = nn.Sequential(*layers)
        
        self.use_vit = use_vit
    
    def forward(self, x):
        return self.encoder(x)


class TTSEnhancer:
    """Main class for enhancing TTS outputs"""
    
    def __init__(self, model_path='best_denoise_encoder.pth', device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        print(f"Loading models on {self.device}...")
        
        # Load KNN-VC model
        self.knn_vc = torch.hub.load('bshall/knn-vc', 'knn_vc', prematched=False, trust_repo=True, pretrained=True)
        
        # Load denoise encoder
        self.denoise_encoder = DenoiseEncoder(
            embedding_dim=1024,  # KNN-VC WavLM embedding dimension
            hidden_dim=1024,
            num_layers=3,
            use_vit=False
        ).to(self.device)
        
        # Load trained weights
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.denoise_encoder.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from epoch {checkpoint['epoch']} with val loss {checkpoint['val_loss']:.4f}")
        else:
            print("Warning: No trained model found, using random initialization")
        
        self.denoise_encoder.eval()
        
        # We'll use a reference audio for the matching set
        self.reference_audio_path = None
    
    def enhance_audio(self, audio_path, return_comparison=True):
        """
        Enhance audio using the denoise encoder
        
        Args:
            audio_path: Path to input audio file
            return_comparison: If True, returns both original and enhanced audio
        
        Returns:
            If return_comparison=True: (original_audio, enhanced_audio, sample_rate)
            Otherwise: (enhanced_audio, sample_rate)
        """
        # Ensure 16kHz audio
        waveform, sr = torchaudio.load(audio_path)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            waveform = resampler(waveform)
            sr = 16000
            
            # Save resampled audio
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                torchaudio.save(tmp.name, waveform, sr)
                audio_path = tmp.name
        
        with torch.no_grad():
            # Extract features (returns 2D tensor: [sequence_length, feature_dim])
            features = self.knn_vc.get_features(audio_path)
            
            # For KNN-VC, we need to use the same audio as reference for self-reconstruction
            # This maintains speaker characteristics while enhancing quality
            matching_set = self.knn_vc.get_matching_set([audio_path])
            
            # Add batch dimension for denoise encoder: [1, sequence_length, feature_dim]
            features_batched = features.unsqueeze(0).to(self.device)
            
            # Denoise features
            denoised_features_batched = self.denoise_encoder(features_batched)
            
            # Remove batch dimension: [sequence_length, feature_dim]
            denoised_features = denoised_features_batched.squeeze(0).cpu()
            
            # Generate audio using KNN matching
            # Using the denoised features as query and original as reference
            enhanced_audio = self.knn_vc.match(denoised_features, matching_set, topk=4)
            enhanced_audio = enhanced_audio.cpu().numpy()
            
            if return_comparison:
                # Generate audio from original features for comparison
                original_recon = self.knn_vc.match(features, matching_set, topk=4)
                original_recon = original_recon.cpu().numpy()
                return original_recon, enhanced_audio, sr
            
            return enhanced_audio, sr
    
    def apply_degradation(self, audio_path, degradation_type='mp3'):
        """Apply degradation to audio for demonstration purposes"""
        waveform, sr = torchaudio.load(audio_path)
        
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            waveform = resampler(waveform)
            sr = 16000
        
        if degradation_type == 'mp3':
            # Low bitrate MP3
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_mp3:
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_wav:
                    torchaudio.save(tmp_wav.name, waveform, sr)
                    subprocess.run([
                        'ffmpeg', '-i', tmp_wav.name, '-b:a', '32k', '-y', tmp_mp3.name
                    ], capture_output=True)
                    subprocess.run([
                        'ffmpeg', '-i', tmp_mp3.name, '-ar', str(sr), '-y', tmp_wav.name
                    ], capture_output=True)
                    degraded, _ = torchaudio.load(tmp_wav.name)
                    os.unlink(tmp_mp3.name)
                    os.unlink(tmp_wav.name)
        
        elif degradation_type == 'mulaw':
            # μ-law encoding
            mulaw_encoded = torchaudio.functional.mu_law_encoding(waveform, 256)
            degraded = torchaudio.functional.mu_law_decoding(mulaw_encoded, 256)
        
        elif degradation_type == 'quantize':
            # Heavy quantization
            degraded = torch.round(waveform * 32) / 32
        
        elif degradation_type == 'bandlimit':
            # Low-pass filter
            degraded = torchaudio.functional.lowpass_biquad(waveform, sr, 4000)
        
        # Save degraded audio
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            torchaudio.save(tmp.name, degraded, sr)
            return tmp.name


# Initialize the enhancer
enhancer = None

def load_model():
    global enhancer
    if enhancer is None:
        enhancer = TTSEnhancer()
    return enhancer


def process_audio(audio_file, mode="enhance", degradation_type="mp3"):
    """
    Process audio based on selected mode
    
    Args:
        audio_file: Input audio file
        mode: "enhance" or "demonstrate"
        degradation_type: Type of degradation to apply in demonstrate mode
    """
    if audio_file is None:
        return None, None, None, "Please upload an audio file"
    
    enhancer = load_model()
    
    try:
        if mode == "enhance":
            # Direct enhancement mode
            original_recon, enhanced_audio, sr = enhancer.enhance_audio(audio_file, return_comparison=True)
            
            # Save outputs
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_orig:
                torchaudio.save(tmp_orig.name, torch.tensor(original_recon).unsqueeze(0), sr)
                original_path = tmp_orig.name
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_enh:
                torchaudio.save(tmp_enh.name, torch.tensor(enhanced_audio).unsqueeze(0), sr)
                enhanced_path = tmp_enh.name
            
            return original_path, enhanced_path, None, "Enhancement complete!"
        
        else:  # demonstrate mode
            # First apply degradation
            degraded_path = enhancer.apply_degradation(audio_file, degradation_type)
            
            # Then enhance the degraded audio - note we only need 2 return values here
            enhanced_audio, sr = enhancer.enhance_audio(degraded_path, return_comparison=False)
            
            # Save enhanced output
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_enh:
                torchaudio.save(tmp_enh.name, torch.tensor(enhanced_audio).unsqueeze(0), sr)
                enhanced_path = tmp_enh.name
            
            return audio_file, degraded_path, enhanced_path, f"Applied {degradation_type} degradation and enhancement"
    
    except Exception as e:
        return None, None, None, f"Error: {str(e)}"


# Create Gradio interface
def create_demo():
    with gr.Blocks(title="TTS Enhancement Demo") as demo:
        gr.Markdown("""
        # 🎙️ TTS Enhancement with Denoise Encoder
        
        This demo showcases a speech enhancement system specifically designed for improving TTS outputs.
        The system uses a lightweight denoise encoder to clean WavLM embeddings before vocoding.
        
        ## How to use:
        1. **Enhance Mode**: Upload any audio to enhance its quality
        2. **Demonstrate Mode**: Apply degradation first, then see how the model recovers the quality
        """)
        
        with gr.Row():
            with gr.Column():
                # Input section
                audio_input = gr.Audio(label="Input Audio", type="filepath")
                
                mode = gr.Radio(
                    choices=["enhance", "demonstrate"], 
                    value="enhance",
                    label="Mode",
                    info="Enhance: Direct enhancement | Demonstrate: Apply degradation then enhance"
                )
                
                degradation_type = gr.Dropdown(
                    choices=["mp3", "mulaw", "quantize", "bandlimit"],
                    value="mp3",
                    label="Degradation Type (for demonstrate mode)",
                    visible=False
                )
                
                process_btn = gr.Button("Process Audio", variant="primary")
                
            with gr.Column():
                # Output section
                status = gr.Textbox(label="Status", value="Ready")
                
                with gr.Tab("Enhance Mode Output"):
                    original_output = gr.Audio(label="Original Reconstruction", type="filepath")
                    enhanced_output = gr.Audio(label="Enhanced Audio", type="filepath")
                
                with gr.Tab("Demonstrate Mode Output"):
                    original_demo = gr.Audio(label="Original Audio", type="filepath")
                    degraded_output = gr.Audio(label="Degraded Audio", type="filepath")
                    enhanced_demo = gr.Audio(label="Enhanced Audio", type="filepath")
        
        # Examples
        gr.Examples(
            examples=[
                ["example_tts.wav", "enhance", "mp3"],
                ["example_speech.wav", "demonstrate", "mulaw"],
            ],
            inputs=[audio_input, mode, degradation_type],
            outputs=[original_output, enhanced_output, status],
            fn=process_audio,
            cache_examples=False
        )
        
        # Event handlers
        def update_visibility(mode):
            return gr.update(visible=(mode == "demonstrate"))
        
        mode.change(fn=update_visibility, inputs=[mode], outputs=[degradation_type])
        
        def process_and_update(audio_file, mode, degradation_type):
            if mode == "enhance":
                orig, enh, _, status = process_audio(audio_file, mode, degradation_type)
                return orig, enh, None, None, None, status
            else:
                orig, deg, enh, status = process_audio(audio_file, mode, degradation_type)
                return None, None, orig, deg, enh, status
        
        process_btn.click(
            fn=process_and_update,
            inputs=[audio_input, mode, degradation_type],
            outputs=[original_output, enhanced_output, original_demo, degraded_output, enhanced_demo, status]
        )
        
        gr.Markdown("""
        ## Technical Details:
        - **Model**: 3-layer MLP denoise encoder (~3M parameters)
        - **Features**: WavLM embeddings (1024-dim)
        - **Vocoder**: HiFi-GAN from KNN-VC
        - **Training**: MSE loss on clean/noisy embedding pairs with masking for variable-length sequences
        
        The model is trained to remove common TTS artifacts like vocoding noise, quantization distortion, and bandwidth limitations.
        """)
    
    return demo


if __name__ == "__main__":
    # Create and launch the demo
    demo = create_demo()
    
    # Launch with public link for easy sharing
    demo.launch(
        server_name="0.0.0.0",  # Allow external connections
        server_port=7860,       # Default Gradio port
        share=False             # Disable share link to avoid connection issues
    )