#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simplified 3D UNet prediction script
Run directly; all configurations are at the top of the script
"""

import torch
import numpy as np
import nibabel as nib
from pathlib import Path
import sys
import warnings
import time
from tqdm import tqdm

warnings.filterwarnings('ignore')


from predictor import UNet3DPredictor

"""
Usage:

1. Batch prediction (default):
   python simple_predict.py

2. Modify model configuration:
   Directly edit the parameters in CONFIG['model_config']:
   - n_channels: number of input channels
   - n_classes: number of output classes  
   - base_channels: base channel size (affects model size)
   - num_groups: number of GroupNorm groups
   - bilinear: upsampling method

3. Common configuration examples:
   # Small model (memory-saving):
   'base_channels': 32, 'num_groups': 4
   
   # Large model (better performance):
   'base_channels': 128, 'num_groups': 16
   
   # Multi-class tasks:
   'n_classes': 5  # e.g., 5 classes

4. Import in other scripts:
   from simple_predict import predict_single_file, predict_images
   predict_single_file("image.nii.gz", "output.nii.gz")

Output files:
- pred_*.nii.gz: prediction results
- prediction_stats.json: statistics (if enabled)

Notes:
- After modifying the model configuration, ensure it matches the training configuration
- If weight loading fails, it will automatically try a relaxed mode
- base_channels must be divisible by num_groups
"""

# ==================== Configuration ====================
CONFIG = {
    # Model and paths
    'model_path': r"D:\unet3d_english\train_end\best_model.pth",  # model weight path
    'input_folder': r"D:\unet3d_english\dataset\test\images",   # input folder
    'output_folder': r"D:\unet3d_english\predict",                # output folder
    
    # Image processing
    'target_size': (64, 64, 64),    # target size, set None to keep original size
    'file_pattern': "*.nii.gz",     # file pattern
    
    # System settings
    'device': 'auto',               # 'auto', 'cpu', 'cuda'
    'save_stats': True,             # save statistics or not
    
    # ========== Model config from training (editable here) ==========
    'model_config': {
        'n_channels': 1,        # number of input channels (1=grayscale, 3=RGB)
        'n_classes': 2,         # number of output classes (2=binary bg+fg, 3=three-class, etc.)
        'base_channels': 64,    # base channel size (affects model capacity: 32/64/128)
        'num_groups': 8,        # number of GroupNorm groups (usually 8 or 16)
        'bilinear': False       # upsampling method (False=transpose conv, True=bilinear)
    }
}

class CustomUNet3DPredictor(UNet3DPredictor):
    """Custom 3D UNet predictor supporting external model configurations"""
    
    def __init__(self, model_path: str, device: str = 'auto', custom_model_config: dict = None):
        """
        Initialize the predictor
        
        Args:
            model_path: path to model weights
            device: computation device
            custom_model_config: custom model configuration
        """
        self.custom_model_config = custom_model_config
        super().__init__(model_path, device)
    
    def _load_model(self):
        """Load model weights (supports custom config)"""
        print(f"Loading model weights: {self.model_path}")
        
        try:
            checkpoint = self.safe_torch_load(self.model_path)
            
            # Extract model configuration from checkpoint
            if 'model_config' in checkpoint:
                saved_model_config = checkpoint['model_config']
                print(f"Model config in checkpoint: {saved_model_config}")
            else:
                saved_model_config = {}
                print("No model config found in checkpoint")
            
            # Use custom config if provided
            if self.custom_model_config:
                print("🔧 Using custom model configuration:")
                self.model_config = self.custom_model_config.copy()
                for key, value in self.model_config.items():
                    print(f"  {key}: {value}")
            else:
                # Otherwise use default logic
                self.model_config = {
                    'n_channels': saved_model_config.get('n_channels', 1),
                    'n_classes': saved_model_config.get('n_classes', 2),
                    'base_channels': saved_model_config.get('base_channels', 64),
                    'num_groups': saved_model_config.get('num_groups', 8),
                    'bilinear': saved_model_config.get('bilinear', False)
                }
                print("Using checkpoint or default model configuration")
            
            print(f"Final model configuration: {self.model_config}")
            
            # Build model
            from src.network_architecture.unet3d import UNet3D
            self.model = UNet3D(
                n_channels=self.model_config['n_channels'],
                n_classes=self.model_config['n_classes'],
                base_channels=self.model_config['base_channels'],
                num_groups=self.model_config['num_groups'],
                bilinear=self.model_config['bilinear']
            ).to(self.device)
            
            # Clean state dict and load weights
            model_state_dict = checkpoint['model_state_dict']
            cleaned_state_dict = self._clean_state_dict(model_state_dict)
            
            # Try loading weights
            try:
                self.model.load_state_dict(cleaned_state_dict, strict=True)
                print("✅ Model weights loaded successfully (strict mode)")
            except RuntimeError as e:
                print(f"⚠️ Strict mode load failed, trying relaxed mode: {str(e)[:100]}...")
                missing_keys, unexpected_keys = self.model.load_state_dict(cleaned_state_dict, strict=False)
                if missing_keys:
                    print(f"Missing keys: {len(missing_keys)}")
                if unexpected_keys:
                    print(f"Unexpected keys: {len(unexpected_keys)}")
                print("⚠️ Model weights loaded (relaxed mode)")
            
            self.model.eval()
            
            # Show model info
            total_params, trainable_params = self.model.get_model_size()
            print(f"Model parameters: {total_params:,} ({trainable_params:,} trainable)")
            
            # Show training info
            if 'epoch' in checkpoint:
                print(f"Model trained up to epoch {checkpoint['epoch']}")
            if 'best_val_dice' in checkpoint:
                print(f"Best validation Dice: {checkpoint['best_val_dice']:.4f}")
            if 'train_loss' in checkpoint:
                print(f"Training loss: {checkpoint['train_loss']:.4f}")
            if 'val_loss' in checkpoint:
                print(f"Validation loss: {checkpoint['val_loss']:.4f}")
                
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Please check the following:")
            print("1. Model file exists and is complete")
            print("2. Model configuration is correct")
            print("3. PyTorch version compatibility")
            raise

def predict_images():
    """Run image prediction"""
    print("3D UNet prediction system started")
    print("=" * 40)
    
    # Show configuration
    print("Configuration:")
    for key, value in CONFIG.items():
        if key == 'model_config':
            print(f"  {key}:")
            for sub_key, sub_value in value.items():
                print(f"    {sub_key}: {sub_value}")
        else:
            print(f"  {key}: {value}")
    print("-" * 40)
    
    try:
        # Initialize custom predictor
        print("Loading model...")
        predictor = CustomUNet3DPredictor(
            model_path=CONFIG['model_path'],
            device=CONFIG['device'],
            custom_model_config=CONFIG['model_config']
        )
        
        # Run batch prediction
        print("Starting prediction...")
        stats = predictor.predict_folder(
            input_folder=CONFIG['input_folder'],
            output_folder=CONFIG['output_folder'],
            target_size=CONFIG['target_size'],
            file_pattern=CONFIG['file_pattern'],
            save_stats=CONFIG['save_stats']
        )
        
        print("\nPrediction statistics:")
        print(f"  Successful predictions: {stats['successful_predictions']} files")
        print(f"  Total time: {stats['total_inference_time']:.2f} seconds")
        print(f"  Average time: {stats['average_inference_time']:.2f} seconds/file")
        
        print(f"\nResults saved to: {CONFIG['output_folder']}")
        print("Prediction completed!")
        
    except FileNotFoundError as e:
        print(f"File or folder not found: {e}")
        print("Please check the following paths:")
        print(f"  Model file: {CONFIG['model_path']}")
        print(f"  Input folder: {CONFIG['input_folder']}")
    
    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()

def predict_single_file(image_path, output_path=None):
    """
    Convenience function to predict a single file
    
    Args:
        image_path: path to input image
        output_path: output path (optional)
    """
    if output_path is None:
        input_path = Path(image_path)
        output_path = input_path.parent / f"pred_{input_path.name}"
    
    try:
        # Initialize custom predictor
        predictor = CustomUNet3DPredictor(
            model_path=CONFIG['model_path'], 
            device=CONFIG['device'],
            custom_model_config=CONFIG['model_config']
        )
        
        # Predict
        prediction, stats = predictor.predict_single_image(
            Path(image_path), 
            CONFIG['target_size']
        )
        
        # Save
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        predictor.save_prediction(prediction, Path(image_path), Path(output_path))
        
        print(f"Prediction completed: {output_path}")
        print(f"Foreground ratio: {stats['foreground_ratio']:.1%}")
        print(f"Inference time: {stats['inference_time']:.2f} seconds")
        
    except Exception as e:
        print(f"Single-file prediction failed: {e}")

def validate_config():
    """Validate configuration parameters"""
    config = CONFIG['model_config']
    
    print("🔍 Validating model configuration...")
    
    # Check parameter ranges
    if config['n_channels'] < 1:
        print("⚠️ Warning: n_channels should be >= 1")
    
    if config['n_classes'] < 2:
        print("⚠️ Warning: n_classes should be >= 2")
    
    if config['base_channels'] not in [16, 32, 64, 128, 256]:
        print(f"⚠️ Warning: base_channels={config['base_channels']} is not a common value (16,32,64,128,256)")
    
    if config['base_channels'] % config['num_groups'] != 0:
        print(f"⚠️ Warning: base_channels({config['base_channels']}) should be divisible by num_groups({config['num_groups']})")
    
    print("✅ Configuration validation completed")

def main():
    """Main function"""
    print("Simplified 3D UNet Prediction Tool")
    print("Supports custom model configuration")
    print()
    
    # Validate configuration
    validate_config()
    print()
    
    # Run batch prediction
    predict_images()

if __name__ == '__main__':
    main()
