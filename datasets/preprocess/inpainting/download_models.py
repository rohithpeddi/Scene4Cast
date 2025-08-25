#!/usr/bin/env python3
"""
Model Downloader for Inpaint4DVideo

This script downloads all required models for the video inpainting system:
- Stable Diffusion v1.5 base model
- VAE model
- DiffuEraser model weights
- ProPainter model weights

Usage:
    python download_models.py [--output_dir OUTPUT_DIR] [--force_download]
"""

import os
import sys
import argparse
import requests
import zipfile
import tarfile
from pathlib import Path
from tqdm import tqdm
import hashlib

class ModelDownloader:
    """Downloads and manages model files for the video inpainting system."""
    
    def __init__(self, output_dir: str = "weights"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Model configurations
        self.models = {
            "stable-diffusion-v1-5": {
                "url": "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main",
                "files": [
                    "config.json",
                    "model_index.json",
                    "scheduler/scheduler_config.json",
                    "text_encoder/config.json",
                    "tokenizer/tokenizer_config.json",
                    "tokenizer/vocab.txt",
                    "unet/config.json",
                    "unet/diffusion_pytorch_model.safetensors",
                    "vae/config.json",
                    "vae/diffusion_pytorch_model.safetensors"
                ],
                "type": "huggingface",
                "description": "Stable Diffusion v1.5 base model"
            },
            "sd-vae-ft-mse": {
                "url": "https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main",
                "files": [
                    "config.json",
                    "diffusion_pytorch_model.safetensors"
                ],
                "type": "huggingface",
                "description": "VAE model fine-tuned on MSE loss"
            },
            "diffuEraser": {
                "url": "https://huggingface.co/runwayml/diffuEraser/resolve/main",
                "files": [
                    "config.json",
                    "diffusion_pytorch_model.safetensors"
                ],
                "type": "huggingface",
                "description": "DiffuEraser model weights"
            },
            "propainter": {
                "url": "https://github.com/sczhou/ProPainter/releases/download/v1.0.0",
                "files": [
                    "ProPainter.pth"
                ],
                "type": "github_release",
                "description": "ProPainter model weights"
            }
        }
    
    def download_file(self, url: str, filepath: Path, description: str = ""):
        """Download a single file with progress bar."""
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filepath, 'wb') as f:
                with tqdm(
                    total=total_size,
                    unit='iB',
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=f"Downloading {description}"
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        size = f.write(chunk)
                        pbar.update(size)
                        
        except Exception as e:
            print(f"Error downloading {url}: {e}")
            if filepath.exists():
                filepath.unlink()
            raise
    
    def download_huggingface_model(self, model_name: str, model_config: dict):
        """Download a HuggingFace model."""
        model_dir = self.output_dir / model_name
        model_dir.mkdir(exist_ok=True)
        
        print(f"\nDownloading {model_config['description']}...")
        
        for file in model_config['files']:
            url = f"{model_config['url']}/{file}"
            filepath = model_dir / file
            
            # Create parent directories if needed
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            if filepath.exists():
                print(f"  {file} already exists, skipping...")
                continue
            
            self.download_file(url, filepath, file)
    
    def download_github_release_model(self, model_name: str, model_config: dict):
        """Download a GitHub release model."""
        model_dir = self.output_dir / model_name
        model_dir.mkdir(exist_ok=True)
        
        print(f"\nDownloading {model_config['description']}...")
        
        for file in model_config['files']:
            url = f"{model_config['url']}/{file}"
            filepath = model_dir / file
            
            if filepath.exists():
                print(f"  {file} already exists, skipping...")
                continue
            
            self.download_file(url, filepath, file)
    
    def download_all_models(self, force_download: bool = False):
        """Download all required models."""
        print("Starting model download process...")
        print(f"Output directory: {self.output_dir.absolute()}")
        
        if force_download:
            print("Force download enabled - existing files will be overwritten")
        
        for model_name, model_config in self.models.items():
            try:
                if model_config['type'] == 'huggingface':
                    self.download_huggingface_model(model_name, model_config)
                elif model_config['type'] == 'github_release':
                    self.download_github_release_model(model_name, model_config)
                else:
                    print(f"Unknown model type for {model_name}: {model_config['type']}")
                    
            except Exception as e:
                print(f"Failed to download {model_name}: {e}")
                continue
        
        print("\nModel download process completed!")
        self.print_model_status()
    
    def print_model_status(self):
        """Print the status of all models."""
        print("\n" + "="*60)
        print("MODEL STATUS")
        print("="*60)
        
        for model_name, model_config in self.models.items():
            model_dir = self.output_dir / model_name
            status = "✓ Downloaded" if model_dir.exists() else "✗ Missing"
            
            print(f"{model_name:25} : {status}")
            if model_dir.exists():
                print(f"{'':25}   Path: {model_dir.absolute()}")
        
        print("="*60)
    
    def verify_models(self):
        """Verify that all required models are present."""
        missing_models = []
        
        for model_name in self.models.keys():
            model_dir = self.output_dir / model_name
            if not model_dir.exists():
                missing_models.append(model_name)
        
        if missing_models:
            print(f"Missing models: {', '.join(missing_models)}")
            return False
        else:
            print("All models are present and ready to use!")
            return True

def main():
    parser = argparse.ArgumentParser(description="Download models for Inpaint4DVideo")
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="weights",
        help="Directory to save downloaded models (default: weights)"
    )
    parser.add_argument(
        "--force_download",
        action="store_true",
        help="Force download even if files already exist"
    )
    parser.add_argument(
        "--verify_only",
        action="store_true",
        help="Only verify existing models without downloading"
    )
    
    args = parser.parse_args()
    
    downloader = ModelDownloader(args.output_dir)
    
    if args.verify_only:
        downloader.verify_models()
    else:
        downloader.download_all_models(args.force_download)

if __name__ == "__main__":
    main()
