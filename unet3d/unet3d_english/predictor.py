#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3D UNet Prediction Script
Load trained model weights and predict nii.gz or nii files in a specified folder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import nibabel as nib
from pathlib import Path
import sys
import argparse
import warnings
import json
from typing import List, Tuple, Optional, Dict
import time
from tqdm import tqdm
import os

# Ignore warnings
warnings.filterwarnings('ignore')

from src.network_architecture.net_module import *
from src.network_architecture.unet3d import UNet3D
from trainer import EnhancedUNet3DTrainer

class UNet3DPredictor:
    """3D UNet Predictor Class"""
    
    def __init__(self, model_path: str, device: str = 'auto'):
        """
        Initialize predictor
        
        Args:
            model_path: Path to model weights (.pth)
            device: Computation device ('auto', 'cpu', 'cuda')
        """
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file does not exist: {model_path}")
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        if self.device.type == 'cuda':
            print(f"GPU name: {torch.cuda.get_device_name(self.device)}")
            print(f"GPU memory: {torch.cuda.get_device_properties(self.device).total_memory / 1024**3:.1f} GB")
        
        # Load model
        self.model = None
        self.model_config = None
        self._load_model()
        
        print("Model loaded, ready for prediction")
    
    def safe_torch_load(self, path):
        """Safe torch.load compatible with PyTorch 2.6+"""
        try:
            return torch.load(path, map_location=self.device, weights_only=False)
        except Exception as e1:
            try:
                print(f"Failed with weights_only=False, trying weights_only=True")
                return torch.load(path, map_location=self.device, weights_only=True)
            except Exception as e2:
                print(f"Setting safe global variables and reloading...")
                torch.serialization.add_safe_globals([
                    'numpy._core.multiarray.scalar',
                    'numpy.core.multiarray.scalar',
                    'numpy.dtype',
                    'numpy.ndarray',
                    'collections.OrderedDict'
                ])
                return torch.load(path, map_location=self.device, weights_only=True)
    
    def _clean_state_dict(self, state_dict):
        """Clean state dict, remove extra keys added by thop"""
        keys_to_remove = []
        for key in state_dict.keys():
            if 'total_ops' in key or 'total_params' in key:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del state_dict[key]
            
        return state_dict
    
    def _load_model(self):
        """Load model weights"""
        print(f"Loading model weights: {self.model_path}")
        
        try:
            checkpoint = self.safe_torch_load(self.model_path)
            
            # Extract model configuration
            if 'model_config' in checkpoint:
                self.model_config = checkpoint['model_config']
                print(f"Loaded model config: {self.model_config}")
            else:
                # Use default config
                self.model_config = {
                    'n_channels': 1,
                    'n_classes': 2,
                    'base_channels': 64,
                    'num_groups': 8,
                    'bilinear': False
                }
                print("Using default model config")
            
            # Build model
            self.model = UNet3D(
                n_channels=self.model_config.get('n_channels', 1),
                n_classes=self.model_config.get('n_classes', 2),
                base_channels=self.model_config.get('base_channels', 64),
                num_groups=self.model_config.get('num_groups', 8),
                bilinear=self.model_config.get('bilinear', False)
            ).to(self.device)
            
            # Clean state dict and load weights
            model_state_dict = checkpoint['model_state_dict']
            cleaned_state_dict = self._clean_state_dict(model_state_dict)
            self.model.load_state_dict(cleaned_state_dict)
            self.model.eval()
            
            # Show model info
            total_params, trainable_params = self.model.get_model_size()
            print(f"Model parameters: {total_params:,} ({trainable_params:,} trainable)")
            
            # Show training info
            if 'epoch' in checkpoint:
                print(f"Model trained up to epoch {checkpoint['epoch']}")
            if 'best_val_dice' in checkpoint:
                print(f"Best validation Dice: {checkpoint['best_val_dice']:.4f}")
                
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def load_nii_image(self, file_path: Path) -> np.ndarray:
        """Load NII.GZ file"""
        try:
            nii_img = nib.load(str(file_path))
            img_data = nii_img.get_fdata()
            img_data = np.array(img_data, dtype=np.float32)
            
            # Handle different dimensions
            if img_data.ndim == 4:
                # If 4D (X, Y, Z, T), take first time point
                print(f"⚠️ Detected 4D image {img_data.shape}, using index=0 of last dimension")
                img_data = img_data[..., 0]
            elif img_data.ndim == 3:
                # 3D image
                print(f"✅ Detected 3D image {img_data.shape}")
            elif img_data.ndim == 2:
                # If 2D, add depth dimension
                print(f"⚠️ Detected 2D image {img_data.shape}, adding depth dimension")
                img_data = img_data[np.newaxis, ...]
            elif img_data.ndim > 4:
                # More than 4D, take first 3 dims
                print(f"⚠️ Detected {img_data.ndim}D image {img_data.shape}, keeping first 3 dims")
                img_data = img_data[..., 0, 0] if img_data.ndim == 5 else img_data
                while img_data.ndim > 3:
                    img_data = img_data[..., 0]
            else:
                raise ValueError(f"Unsupported image dimension: {img_data.ndim}D")
            
            # Ensure final image is 3D
            if img_data.ndim != 3:
                raise ValueError(f"Processed image dimension incorrect: {img_data.ndim}D, expected 3D")
                
            print(f"📊 Final image shape: {img_data.shape}")
            return img_data
            
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            raise
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image"""
        # Clip outliers
        p1, p99 = np.percentile(image, (1, 99))
        image = np.clip(image, p1, p99)
        
        # Z-score normalization
        mean = np.mean(image)
        std = np.std(image)
        if std > 0:
            image = (image - mean) / std
        
        return image
    
    def resize_volume(self, volume: np.ndarray, target_size: Tuple[int, int, int]) -> np.ndarray:
        """Resize 3D volume"""
        if target_size is None:
            return volume
            
        from scipy.ndimage import zoom
        
        current_size = volume.shape
        zoom_factors = [t/c for t, c in zip(target_size, current_size)]
        resized = zoom(volume, zoom_factors, order=1)  # linear interpolation
        
        return resized
    
    def preprocess_image(self, image: np.ndarray, target_size: Optional[Tuple[int, int, int]] = None) -> torch.Tensor:
        """Preprocess a single image"""
        # Resize
        if target_size is not None:
            image = self.resize_volume(image, target_size)
        
        # Normalize
        image = self.normalize_image(image)
        
        # Add channel and batch dims: [D,H,W] -> [1,1,D,H,W]
        image = image[np.newaxis, np.newaxis, ...]
        
        # Convert to tensor
        image = torch.from_numpy(image).float().to(self.device)
        
        return image
    
    def postprocess_prediction(self, prediction: torch.Tensor, original_size: Tuple[int, int, int]) -> np.ndarray:
        """Post-process prediction"""
        # Remove batch and channel dims
        if len(prediction.shape) == 5:
            prediction = prediction.squeeze(0)
        
        # Multi-class: take argmax
        if prediction.shape[0] > 1:
            prediction = torch.argmax(prediction, dim=0)
        else:
            # Binary classification
            prediction = torch.sigmoid(prediction.squeeze(0))
            prediction = (prediction > 0.5).float()
        
        prediction = prediction.cpu().numpy().astype(np.uint8)
        
        # Resize back to original size
        if prediction.shape != original_size:
            from scipy.ndimage import zoom
            zoom_factors = [o/p for o, p in zip(original_size, prediction.shape)]
            prediction = zoom(prediction, zoom_factors, order=0)
            prediction = prediction.astype(np.uint8)
        
        return prediction
    
    def predict_single_image(self, image_path: Path, target_size: Optional[Tuple[int, int, int]] = None) -> Tuple[np.ndarray, Dict]:
        """Predict a single image"""
        original_image = self.load_nii_image(image_path)
        original_size = original_image.shape
        
        input_tensor = self.preprocess_image(original_image, target_size)
        
        with torch.no_grad():
            start_time = time.time()
            prediction = self.model(input_tensor)
            inference_time = time.time() - start_time
        
        prediction_mask = self.postprocess_prediction(prediction, original_size)
        
        stats = {
            'original_size': original_size,
            'input_size': input_tensor.shape[2:],
            'inference_time': inference_time,
            'foreground_pixels': int(np.sum(prediction_mask > 0)),
            'total_pixels': int(prediction_mask.size),
            'foreground_ratio': float(np.sum(prediction_mask > 0) / prediction_mask.size)
        }
        
        return prediction_mask, stats
    
    def save_prediction(self, prediction: np.ndarray, original_nii_path: Path, output_path: Path):
        """Save prediction as NII.GZ"""
        try:
            original_nii = nib.load(str(original_nii_path))
            prediction_nii = nib.Nifti1Image(
                prediction,
                original_nii.affine,
                original_nii.header
            )
            prediction_nii.set_data_dtype(np.uint8)
            nib.save(prediction_nii, str(output_path))
            print(f"Prediction saved: {output_path}")
        except Exception as e:
            print(f"Error saving prediction: {e}")
            raise
    
    def predict_folder(self, 
                      input_folder: str, 
                      output_folder: str, 
                      target_size: Optional[Tuple[int, int, int]] = None,
                      file_pattern: str = "*.nii.gz",
                      save_stats: bool = True) -> Dict:
        """
        Predict all NII.GZ files in a folder
        """
        input_path = Path(input_folder)
        output_path = Path(output_folder)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input folder does not exist: {input_path}")
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        nii_files = list(input_path.glob(file_pattern))
        if not nii_files:
            raise ValueError(f"No files matching {file_pattern} found in {input_path}")
        
        print(f"Found {len(nii_files)} files to predict")
        print(f"Output folder: {output_path}")
        if target_size:
            print(f"Target size: {target_size}")
        
        all_stats = {}
        total_time = 0
        successful_predictions = 0
        
        for nii_file in tqdm(nii_files, desc="Prediction progress"):
            try:
                prediction, stats = self.predict_single_image(nii_file, target_size)
                output_file = output_path / f"pred_{nii_file.name}"
                self.save_prediction(prediction, nii_file, output_file)
                
                stats['input_file'] = str(nii_file)
                stats['output_file'] = str(output_file)
                all_stats[nii_file.name] = stats
                
                total_time += stats['inference_time']
                successful_predictions += 1
                
                tqdm.write(f"✅ {nii_file.name}: {stats['foreground_ratio']:.1%} foreground, "
                           f"{stats['inference_time']:.2f}s")
                
            except Exception as e:
                tqdm.write(f"❌ Prediction failed for {nii_file.name}: {e}")
                continue
        
        summary_stats = {
            'total_files': len(nii_files),
            'successful_predictions': successful_predictions,
            'failed_predictions': len(nii_files) - successful_predictions,
            'total_inference_time': total_time,
            'average_inference_time': total_time / successful_predictions if successful_predictions > 0 else 0,
            'model_config': self.model_config,
            'target_size': target_size
        }
        
        print(f"\nPrediction completed!")
        print(f"Successful: {successful_predictions}/{len(nii_files)}")
        print(f"Total inference time: {total_time:.2f} s")
        print(f"Average inference time: {summary_stats['average_inference_time']:.2f} s/file")
        
        if save_stats:
            stats_file = output_path / 'prediction_stats.json'
            full_stats = {
                'summary': summary_stats,
                'individual_files': all_stats
            }
            
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(full_stats, f, ensure_ascii=False, indent=2)
            
            print(f"Statistics saved: {stats_file}")
        
        return summary_stats

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='3D UNet Prediction Script',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('model_path', type=str, help='Path to model weights (.pth)')
    parser.add_argument('input_folder', type=str, help='Path to input folder')
    parser.add_argument('output_folder', type=str, help='Path to output folder')
    
    parser.add_argument('--target_size', type=int, nargs=3, metavar=('D', 'H', 'W'),
                       help='Target size (depth height width), e.g., --target_size 64 64 64')
    parser.add_argument('--device', type=str, default='auto', 
                       choices=['auto', 'cpu', 'cuda'], help='Computation device')
    parser.add_argument('--pattern', type=str, default='*.nii.gz',
                       help='File pattern')
    parser.add_argument('--no_stats', action='store_true',
                       help='Do not save statistics')
    
    args = parser.parse_args()
    
    print("🧠 3D UNet Prediction System")
    print("=" * 50)
    
    try:
        predictor = UNet3DPredictor(args.model_path, args.device)
        target_size = tuple(args.target_size) if args.target_size else None
        
        stats = predictor.predict_folder(
            input_folder=args.input_folder,
            output_folder=args.output_folder,
            target_size=target_size,
            file_pattern=args.pattern,
            save_stats=not args.no_stats
        )
        
        print("\n🎉 All predictions completed!")
        
    except KeyboardInterrupt:
        print("\n⏹️ Prediction interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during prediction: {e}")
        import traceback
        traceback.print_exc()

def predict_single_example():
    """Single file prediction example"""
    print("Single file prediction example:")
    
    model_path = r"model.pth"
    image_path = r"image.nii.gz"
    output_path = r"predictions\pred_image.nii.gz"
    target_size = (64, 64, 64)  # or None to keep original size
    
    try:
        predictor = UNet3DPredictor(model_path, device='auto')
        prediction, stats = predictor.predict_single_image(Path(image_path), target_size)
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        predictor.save_prediction(prediction, Path(image_path), Path(output_path))
        
        print(f"Prediction stats: {stats}")
        print("Prediction completed!")
        
    except Exception as e:
        print(f"Prediction failed: {e}")

if __name__ == "__main__":
    main()
