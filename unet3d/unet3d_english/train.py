#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3D UNet Simplified Training Script - with Warmup and Advanced Optimizer Settings
Only need to adjust parameters, all training functions are in trainer
"""

import torch
import sys
import argparse
from pathlib import Path
import random
import numpy as np
import math

from src.network_architecture.net_module import *  # UNet3D model definition
from trainer import *  # Training tools and trainer
from src.network_architecture.unet3d import UNet3D
from src.data_processing_and_data_enhancement.dataload import MedicalImageDataset, create_data_loaders

"""
🚀 Enhanced 3D UNet Training Script - Integrated Visualization and Warmup

New Advanced Features:
✅ Warmup Learning Rate Scheduling - Gradual learning rate increase
✅ Momentum Warmup - Progressive momentum adjustment
✅ Bias Parameter Special Handling - Different learning rate settings
✅ Parameter Grouping Optimization - Separate handling for weights and biases
✅ Multiple Learning Rate Schedulers (Step, ReduceLROnPlateau, Cosine)
✅ Advanced Optimizer Support (Adam, AdamW, SGD with momentum)

Warmup Mechanism Explanation:
🔥 First N epochs use lower learning rate and momentum, gradually increasing to target values
🌡️  Helps stabilize early training, prevents gradient explosion
⚖️  Bias parameters use different warmup strategy

Recommended Parameter Ranges:
📈 momentum: 0.3-0.98 (SGD), 0.6-0.98 (Adam beta1)
🔥 warmup_epochs: 1-5 epochs (can be decimal, e.g. 2.5)
🌡️  warmup_momentum: 0.0-0.95 (initial momentum)
⚖️  warmup_bias_lr: 0.0-0.2 (bias learning rate multiplier)

Usage Instructions:
1. Basic usage (with warmup):
   python train.py

2. Custom warmup parameters:
   python train.py --warmup_epochs 3.0 --warmup_momentum 0.1 --momentum 0.9

3. Disable warmup:
   python train.py --warmup_epochs 0

4. Use different scheduler:
   python train.py --scheduler cosine

5. Full advanced settings:
   python train.py --epochs 100 --lr 1e-3 --momentum 0.95 \
                   --warmup_epochs 2.5 --warmup_momentum 0.1 \
                   --warmup_bias_lr 0.15 --scheduler cosine

New Output Content:
📈 Warmup learning rate and momentum change curves
📊 Learning rate tracking for different parameter groups
🔥 Detailed logs for warmup phase

Training Phase Explanation:
1️⃣ Warmup Phase (0 - warmup_epochs):
   - Learning rate linearly increases from 0 to target value
   - Momentum increases from warmup_momentum to target value
   - Bias parameters use special learning rate settings

2️⃣ Normal Training Phase (warmup_epochs - total epochs):
   - Use target learning rate and momentum
   - Adjust learning rate according to configured scheduler

Effects:
✅ More stable training start
✅ Better convergence performance
✅ Reduced instability in early training
✅ Support for larger learning rate settings

Notes:
⚠️  Warmup slightly extends training time
⚠️  Need to adjust warmup_epochs based on dataset size
⚠️  Some small datasets may not need warmup
⚠️  Optimal momentum settings differ for SGD and Adam
"""


# ==================== Warmup Learning Rate Scheduler ====================
class WarmupLRScheduler:
    """
    Warmup Learning Rate Scheduler
    Supports warmup for momentum and bias lr
    """
    def __init__(self, optimizer, warmup_epochs, warmup_momentum, warmup_bias_lr, base_lr, base_momentum):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.warmup_momentum = warmup_momentum
        self.warmup_bias_lr = warmup_bias_lr
        self.base_lr = base_lr
        self.base_momentum = base_momentum
        self.current_epoch = 0
        
        # Save original parameter group settings
        self.param_groups_info = []
        for i, param_group in enumerate(optimizer.param_groups):
            info = {
                'base_lr': param_group.get('lr', base_lr),
                'base_momentum': param_group.get('momentum', base_momentum),
                'is_bias': 'bias' in str(param_group.get('name', '')).lower()
            }
            self.param_groups_info.append(info)
    
    def step(self, epoch=None):
        """Update learning rate and momentum"""
        if epoch is not None:
            self.current_epoch = epoch
        else:
            self.current_epoch += 1
        
        if self.current_epoch < self.warmup_epochs:
            # Warmup phase - Fix: Start from step 1, not 0
            warmup_ratio = (self.current_epoch + 1) / self.warmup_epochs
            
            for i, param_group in enumerate(self.optimizer.param_groups):
                info = self.param_groups_info[i]
                
                # Learning rate warmup (different handling for bias and regular parameters)
                if info['is_bias']:
                    # Bias parameters use special warmup
                    target_lr = self.warmup_bias_lr * info['base_lr']
                    param_group['lr'] = target_lr * warmup_ratio
                else:
                    # Linear warmup for regular parameters
                    param_group['lr'] = info['base_lr'] * warmup_ratio
                
                # Momentum warmup (if optimizer supports it)
                if 'momentum' in param_group:
                    param_group['momentum'] = self.warmup_momentum + (info['base_momentum'] - self.warmup_momentum) * warmup_ratio
                elif 'betas' in param_group:  # Adam series
                    # For Adam, adjust beta1 (equivalent to momentum)
                    original_betas = param_group['betas']
                    new_beta1 = self.warmup_momentum + (info['base_momentum'] - self.warmup_momentum) * warmup_ratio
                    param_group['betas'] = (new_beta1, original_betas[1])
        
        else:
            # Warmup complete, restore normal settings
            for i, param_group in enumerate(self.optimizer.param_groups):
                info = self.param_groups_info[i]
                param_group['lr'] = info['base_lr']
                
                if 'momentum' in param_group:
                    param_group['momentum'] = info['base_momentum']
                elif 'betas' in param_group:
                    param_group['betas'] = (info['base_momentum'], param_group['betas'][1])
    
    def get_lr(self):
        """Get current learning rate"""
        return [param_group['lr'] for param_group in self.optimizer.param_groups]
    
    def get_momentum(self):
        """Get current momentum"""
        momenta = []
        for param_group in self.optimizer.param_groups:
            if 'momentum' in param_group:
                momenta.append(param_group['momentum'])
            elif 'betas' in param_group:
                momenta.append(param_group['betas'][0])
            else:
                momenta.append(0.0)
        return momenta
    
    def get_lr(self):
        """Get current learning rate"""
        return [param_group['lr'] for param_group in self.optimizer.param_groups]
    
    def get_momentum(self):
        """Get current momentum"""
        momenta = []
        for param_group in self.optimizer.param_groups:
            if 'momentum' in param_group:
                momenta.append(param_group['momentum'])
            elif 'betas' in param_group:
                momenta.append(param_group['betas'][0])
            else:
                momenta.append(0.0)
        return momenta


# ==================== Training Parameter Configuration ====================
def get_config():
    """
    🔧 Adjust all training parameters here
    """
    config = {
        # === Data Related Parameters ===
        'data_root': r"D:\unet3d_english\dataset",  # 🔧 Change to your data root directory
        'target_size': (64, 64, 64),     # 🔧 Target image size (D, H, W)
        'batch_size': 8,                 # 🔧 Batch size (adjust based on GPU memory)
        'num_workers': 4,                # 🔧 Number of data loading threads
        'use_augmentation': False,        # 🔧 Enable data augmentation (training set only)
        'augmentation_type': 'custom',    # 🔧 Data augmentation type ('light', 'medium', 'heavy', 'medical', 'medical_heavy', 'custom')
        
        # === Model Related Parameters ===
        'n_channels': 1,                 # 🔧 Input channels (1 for grayscale, 3 for RGB)
        'n_classes': 2,                  # 🔧 Output classes (2 for binary, adjust for multi-class)
        'base_channels': 64,             # 🔧 Base number of channels (16, 32, or 64)
        'num_groups': 8,                 # 🔧 GroupNorm number of groups
        'bilinear': False,               # 🔧 Use bilinear upsampling
        
        # === Training Related Parameters ===
        'num_epochs': 300,               # 🔧 Number of training epochs
        'learning_rate': 1e-3,           # 🔧 Learning rate (SGD=1E-2, Adam=1E-3)
        'weight_decay': 5e-4,            # 🔧 Weight decay
        'optimizer': 'adam',             # 🔧 Optimizer ('adam', 'adamw', 'sgd')
        
        # === Advanced Optimizer Parameters ===
        'momentum': 0.937,               # 🔧 SGD momentum or Adam beta1 (0.3, 0.6, 0.98)
        'warmup_epochs': 3.0,            # 🔧 Warmup epochs (supports decimal)
        'warmup_momentum': 0.8,          # 🔧 Warmup initial momentum (0.0, 0.95)
        'warmup_bias_lr': 0.1,           # 🔧 Warmup initial bias lr multiplier (0.0, 0.2)
        
        # === Visualization Related Parameters ===
        'enable_visualization': True,    # 🔧 Enable visualization
        'plot_interval': 20,             # 🔧 Visualization update interval (every N epochs)
        'early_stopping_patience': 31,   # 🔧 Early stopping patience value
        
        # === Loss Function Parameters ===
        'loss_type': 'combined',         # 🔧 Loss function type ('dice', 'ce', 'combined')
        'ce_weight': 0.4,                # 🔧 Cross Entropy weight
        'dice_weight': 0.6,              # 🔧 Dice Loss weight
        
        # === Learning Rate Scheduler ===
        'scheduler': 'reduce_on_plateau', # 🔧 Scheduler type ('step', 'reduce_on_plateau', 'cosine', None)
        'scheduler_patience': 10,         # 🔧 ReduceLROnPlateau patience value
        'scheduler_factor': 0.5,          # 🔧 Learning rate decay factor
        'step_size': 10,                  # 🔧 StepLR step size
        'cosine_t_max': None,             # 🔧 Cosine scheduler max period (None uses total epochs)
        
        # === Save and Logging ===
        'save_dir': r"D:\unet3d_english\train_end",     # 🔧 Model save directory
        'log_interval': 1,               # 🔧 Log output interval
        'save_interval': 200,            # 🔧 Model save interval
        'resume_from': None,             # 🔧 Resume training from checkpoint (path or None)
        
        # === Other Settings ===
        'seed': 42,                      # 🔧 Random seed
        'device': 'auto',                # 🔧 Device selection ('auto', 'cpu', 'cuda:0')
        'run_test': True,                # 🔧 Run test after training completion
    }
    return config

def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def validate_config(config):
    """Validate configuration parameters"""
    print("🔍 Validating configuration parameters...")
    
    data_root = Path(config['data_root'])
    if not data_root.exists():
        raise FileNotFoundError(f"❌ Data directory does not exist: {data_root}")
    
    train_images = data_root / 'train' / 'images'
    train_labels = data_root / 'train' / 'labels'
    
    if not train_images.exists():
        raise FileNotFoundError(f"❌ Training images directory does not exist: {train_images}")
    if not train_labels.exists():
        raise FileNotFoundError(f"❌ Training labels directory does not exist: {train_labels}")
    
    print("✅ Basic configuration validation passed")
    
    for split in ['val', 'test']:
        images_dir = data_root / split / 'images'
        labels_dir = data_root / split / 'labels'
        if images_dir.exists() and labels_dir.exists():
            print(f"✅ Found {split} dataset")
        else:
            print(f"⚠️  {split} dataset is incomplete or does not exist")
    
    # Validate newly added parameters
    if not (0.0 <= config['momentum'] <= 1.0):
        print(f"⚠️  Warning: momentum value {config['momentum']} exceeds recommended range [0.0, 1.0]")
    
    if config['warmup_epochs'] < 0:
        print(f"❌ Error: warmup_epochs cannot be negative: {config['warmup_epochs']}")
        raise ValueError("warmup_epochs must be >= 0")
    
    if not (0.0 <= config['warmup_momentum'] <= 1.0):
        print(f"⚠️  Warning: warmup_momentum value {config['warmup_momentum']} exceeds recommended range [0.0, 1.0]")
    
    if not (0.0 <= config['warmup_bias_lr'] <= 1.0):
        print(f"⚠️  Warning: warmup_bias_lr value {config['warmup_bias_lr']} exceeds recommended range [0.0, 1.0]")
    
    print("🔍 Performing data integrity check...")
    check_data_integrity(config)

def check_data_integrity(config):
    """Check data integrity and label value range"""
    try:
        test_dataset = MedicalImageDataset(
            data_root=config['data_root'],
            split='train',
            target_size=config['target_size'],
            use_augmentation=config['use_augmentation'],
            augmentation_type=config['augmentation_type']
        )
        
        if len(test_dataset) > 0:
            sample = test_dataset[0]
            image = sample['image']
            mask = sample['mask']
            
            print(f"📊 Image shape: {image.shape}")
            print(f"📊 Label shape: {mask.shape}")
            print(f"📊 Image value range: [{image.min().item():.3f}, {image.max().item():.3f}]")
            print(f"📊 Label value range: [{mask.min().item()}, {mask.max().item()}]")
            print(f"📊 Unique label values: {torch.unique(mask).tolist()}")
            
            max_label = mask.max().item()
            if max_label >= config['n_classes']:
                print(f"⚠️  Warning: Maximum label value {max_label} exceeds number of classes {config['n_classes']}")
                print(f"💡 Suggestion: Set n_classes to {max_label + 1} or check label data")
            
            min_label = mask.min().item()
            if min_label < 0:
                print(f"❌ Error: Found negative label value {min_label}")
                raise ValueError("Label values cannot be negative")
            
            print("✅ Data integrity check passed")
        else:
            print("❌ No training data found")
            
    except Exception as e:
        print(f"⚠️  Data integrity check failed: {e}")
        print("💡 Suggestion: Continue execution, but be aware of potential issues")

def print_config_summary(config):
    """Display configuration summary"""
    print("\n" + "=" * 60)
    print("📋 Training Configuration Summary")
    print("=" * 60)
    print(f"📁 Data root directory: {config['data_root']}")
    print(f"🎯 Image size: {config['target_size']}")
    print(f"📦 Batch size: {config['batch_size']}")
    print(f"🏗️  Model channels: {config['n_channels']} -> {config['n_classes']}")
    print(f"🎮 Training epochs: {config['num_epochs']}")
    print(f"⚡ Learning rate: {config['learning_rate']}")
    print(f"🚀 Optimizer: {config['optimizer'].upper()}")
    print(f"📈 Momentum: {config['momentum']}")
    print(f"🔥 Warmup epochs: {config['warmup_epochs']}")
    print(f"🌡️  Warmup momentum: {config['warmup_momentum']}")
    print(f"⚖️  Warmup bias lr: {config['warmup_bias_lr']}")
    print(f"🎯 Loss function: {config['loss_type']}")
    print(f"📅 Scheduler: {config['scheduler']}")
    print(f"💾 Save directory: {config['save_dir']}")
    print(f"🎨 Visualization: {'Enabled' if config['enable_visualization'] else 'Disabled'}")
    if config['enable_visualization']:
        print(f"📊 Visualization interval: Every {config['plot_interval']} epochs")
    print(f"🔄 Data augmentation: {'Enabled' if config['use_augmentation'] else 'Disabled'}")
    if config['use_augmentation']:
        print(f"🎭 Augmentation type: {config['augmentation_type']}")
    print("=" * 60)

def setup_optimizer_with_param_groups(model, config):
    """
    Set up optimizer with support for different parameter groups
    Configure different learning rates for bias parameters and other parameters
    """
    # Separate bias parameters from other parameters
    bias_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'bias' in name.lower():
                bias_params.append(param)
            else:
                other_params.append(param)
    
    # Create parameter groups
    param_groups = [
        {
            'params': other_params,
            'lr': config['learning_rate'],
            'weight_decay': config['weight_decay'],
            'name': 'weights'
        },
        {
            'params': bias_params,
            'lr': config['learning_rate'],
            'weight_decay': 0.0,  # bias usually doesn't use weight decay
            'name': 'bias'
        }
    ]
    
    print(f"📊 Parameter grouping: {len(other_params)} weight parameters, {len(bias_params)} bias parameters")
    
    # Configure based on optimizer type
    if config['optimizer'].lower() == 'adam':
        optimizer = torch.optim.Adam(
            param_groups,
            betas=(config['momentum'], 0.999)  # Use momentum as beta1
        )
    elif config['optimizer'].lower() == 'adamw':
        optimizer = torch.optim.AdamW(
            param_groups,
            betas=(config['momentum'], 0.999)  # Use momentum as beta1
        )
    elif config['optimizer'].lower() == 'sgd':
        # Add momentum parameter for SGD
        for group in param_groups:
            group['momentum'] = config['momentum']
        
        optimizer = torch.optim.SGD(param_groups)
    else:
        raise ValueError(f"Unsupported optimizer: {config['optimizer']}")
    
    return optimizer

def create_lr_scheduler(optimizer, config, warmup_scheduler=None):
    """Create learning rate scheduler (used after warmup)"""
    if not config['scheduler']:
        return None
    
    if config['scheduler'] == 'step':
        from torch.optim.lr_scheduler import StepLR
        scheduler = StepLR(
            optimizer,
            step_size=config['step_size'],
            gamma=config['scheduler_factor']
        )
        print(f"📅 Main scheduler: StepLR (decay by {config['scheduler_factor']} every {config['step_size']} epochs)")
        
    elif config['scheduler'] == 'reduce_on_plateau':
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='max',
            patience=config['scheduler_patience'],
            factor=config['scheduler_factor'],
        )
        print(f"📉 Main scheduler: ReduceLROnPlateau (patience: {config['scheduler_patience']})")
        
    elif config['scheduler'] == 'cosine':
        from torch.optim.lr_scheduler import CosineAnnealingLR
        t_max = config.get('cosine_t_max', config['num_epochs'])
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=t_max,
            eta_min=config['learning_rate'] * 0.01  # Minimum learning rate is 1% of initial value
        )
        print(f"🌊 Main scheduler: CosineAnnealingLR (T_max: {t_max})")
        
    else:
        raise ValueError(f"Unsupported scheduler: {config['scheduler']}")
    
    return scheduler

def create_trainer_from_config(config):
    """Create trainer based on configuration"""
    print("🏗️  Creating enhanced trainer from configuration...")
    
    # Set device
    if config['device'] == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(config['device'])
    
    print(f"🖥️  Using device: {device}")
    if device.type == 'cuda':
        print(f"🔢 GPU name: {torch.cuda.get_device_name(device)}")
        print(f"💾 GPU memory: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f} GB")
    
    # Create model
    print("🏗️  Building model...")
    model = UNet3D(
        n_channels=config['n_channels'],
        n_classes=config['n_classes'],
        base_channels=config['base_channels'],
        num_groups=config['num_groups'],
        bilinear=config['bilinear']
    ).to(device)
    
    total_params, trainable_params = model.get_model_size()
    total_params, trainable_params = model.get_model_size()
    print(f"📊 Model parameters: {total_params:,} ({trainable_params:,} trainable)")
    print(f"💾 Estimated size: {total_params * 4 / 1024 / 1024:.1f} MB (FP32)")

    # Calculate model computational cost
    try:
        from thop import profile
        sample_input = torch.randn(1, config['n_channels'], *config['target_size']).to(device)
        flops, _ = profile(model, inputs=(sample_input,), verbose=False)
        # Divide by 2 to correct for double counting (MAC vs FLOPs)
        flops = flops / 2 / 1e9  # Convert to GFLOPs
        print(f"🔢 Model FLOPs: {flops:.3f}G")
        del sample_input
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    except:
        print("⚠️ Unable to calculate FLOPs (requires thop: pip install thop)")
     
    # Create data loaders (using data augmentation parameters from config)
    print("📁 Building data loaders...")
    data_loaders = create_data_loaders(
        data_root=config['data_root'],
        batch_size=config['batch_size'],
        target_size=config['target_size'],
        num_workers=config['num_workers'],
        use_augmentation=config['use_augmentation'],
        augmentation_type=config['augmentation_type']
    )
    
    # Set up optimizer (with parameter grouping support)
    print("⚡ Setting up advanced optimizer...")
    optimizer = setup_optimizer_with_param_groups(model, config)
    print(f"⚡ Optimizer: {config['optimizer'].upper()}")
    print(f"📈 Base learning rate: {config['learning_rate']}")
    print(f"📈 Momentum/Beta1: {config['momentum']}")
    
    # Set up Warmup scheduler
    warmup_scheduler = None
    if config['warmup_epochs'] > 0:
        warmup_scheduler = WarmupLRScheduler(
            optimizer=optimizer,
            warmup_epochs=config['warmup_epochs'],
            warmup_momentum=config['warmup_momentum'],
            warmup_bias_lr=config['warmup_bias_lr'],
            base_lr=config['learning_rate'],
            base_momentum=config['momentum']
        )
        print(f"🔥 Warmup scheduler: {config['warmup_epochs']} epochs")
        print(f"🌡️  Warmup momentum: {config['warmup_momentum']} -> {config['momentum']}")
        print(f"⚖️  Warmup bias lr multiplier: {config['warmup_bias_lr']}")
    
    # Set up main learning rate scheduler
    main_scheduler = create_lr_scheduler(optimizer, config, warmup_scheduler)
    
    # Set up loss function
    if config['loss_type'] == 'dice':
        criterion = DiceLoss()
        print("🎯 Loss function: Dice Loss")
        
    elif config['loss_type'] == 'ce':
        criterion = torch.nn.CrossEntropyLoss()
        print("🎯 Loss function: Cross Entropy Loss")
        
    elif config['loss_type'] == 'combined':
        criterion = CombinedLoss(
            ce_weight=config['ce_weight'],
            dice_weight=config['dice_weight']
        )
        print(f"🎯 Loss function: Combined Loss (CE: {config['ce_weight']}, Dice: {config['dice_weight']})")
        
    else:
        raise ValueError(f"Unsupported loss function: {config['loss_type']}")
    
    # Create enhanced trainer and pass complete configuration
    trainer = EnhancedUNet3DTrainer(
        model=model,
        data_loaders=data_loaders,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        save_dir=config['save_dir'],
        log_interval=config['log_interval'],
        save_interval=config['save_interval'],
        scheduler=main_scheduler,
        visualize=config['enable_visualization'],
        plot_interval=config['plot_interval'],
        training_config=config  # 🔑 Key: Pass complete configuration
    )
    
    # Add warmup scheduler to trainer
    if warmup_scheduler:
        trainer.warmup_scheduler = warmup_scheduler
        trainer.warmup_epochs = config['warmup_epochs']
        print("✅ Warmup scheduler integrated into trainer")
    
    return trainer

def main():
    """Main training function"""
    print("🚀 Enhanced 3D UNet Training System Starting (with Warmup)")
    print("=" * 60)
    
    # Set CUDA debug mode
    import os
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    print("🔧 CUDA debug mode enabled")
    
    config = get_config()
    
    set_seed(config['seed'])
    print(f"🎲 Random seed set: {config['seed']}")
    
    validate_config(config)
    print_config_summary(config)
    
    trainer = create_trainer_from_config(config)
    
    print("\n🚀 Starting enhanced training (with Warmup)...")
    print("=" * 70)
    
    try:
        # Execute training (including visualization and early stopping)
        trainer.train(
            num_epochs=config['num_epochs'],
            resume_from=config['resume_from'],
            early_stopping_patience=config['early_stopping_patience']
        )
        
        # Test best model after training completion
        if config['run_test'] and 'test' in trainer.data_loaders:
            print("\n" + "=" * 70)
            print("Starting test on best model...")
            best_model_path = Path(config['save_dir']) / 'best_model.pth'
            if best_model_path.exists():
                # Execute test and automatically save results
                test_results = trainer.test(str(best_model_path), save_results=True)
                
                if test_results:
                    # Display results in console
                    print(f"\nFinal test results:")
                    print(f"Average loss: {test_results['avg_loss']:.4f}")
                    print(f"Average Dice score: {test_results['avg_dice']:.4f}")
                    print(f"Dice score percentage: {test_results['avg_dice'] * 100:.2f}%")
                
            else:
                print("Best model file not found")
        
        print("\n" + "🎉" * 20)
        print("✅ All tasks completed!")
        print("📊 Training curves and visualization results saved to: " + str(Path(config['save_dir']) / 'visualizations'))
        print("🎉" * 20)
        
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
        print("💾 Saving interrupted state...")
        try:
            trainer.save_checkpoint(len(trainer.history['train_loss']), is_best=False)
            print("✅ State saved, can resume training using resume_from parameter")
        except:
            print("❌ Failed to save state")
        
    except Exception as e:
        print(f"\n❌ Error occurred during training: {e}")
        import traceback
        traceback.print_exc()
        print("\n💡 Suggestions:")
        print("   - Check if label values are in correct range [0, n_classes-1]")
        print("   - Check if data format is correct")
        print("   - Check if GPU memory is sufficient (can reduce batch_size)")
        print("   - Check if file paths are correct")
        print("   - Check for NaN or abnormal values in data")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Enhanced 3D UNet Training Script - Integrated Visualization and Warmup',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--data_root', type=str, help='Data root directory')
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--resume', type=str, help='Resume training from checkpoint')
    parser.add_argument('--save_dir', type=str, help='Model save directory')
    parser.add_argument('--device', type=str, help='Compute device')
    parser.add_argument('--n_classes', type=int, help='Number of output classes')
    parser.add_argument('--loss_type', type=str, choices=['dice', 'ce', 'combined'], help='Loss function type')
    parser.add_argument('--no_viz', action='store_true', help='Disable visualization')
    parser.add_argument('--plot_interval', type=int, help='Visualization update interval')
    parser.add_argument('--no_augmentation', action='store_true', help='Disable data augmentation')
    parser.add_argument('--augmentation_type', type=str, 
                        choices=['light', 'medium', 'heavy', 'medical', 'medical_heavy'],
                        help='Data augmentation type')
    
    # Add Warmup related parameters
    parser.add_argument('--momentum', type=float, help='SGD momentum or Adam beta1')
    parser.add_argument('--warmup_epochs', type=float, help='Number of warmup epochs (supports decimal)')
    parser.add_argument('--warmup_momentum', type=float, help='Warmup initial momentum')
    parser.add_argument('--warmup_bias_lr', type=float, help='Warmup bias learning rate multiplier')
    parser.add_argument('--scheduler', type=str, 
                        choices=['step', 'reduce_on_plateau', 'cosine', None],
                        help='Learning rate scheduler type')
    
    return parser.parse_args()

def update_config_from_args(config, args):
    """Update configuration using command line arguments"""
    updated = False
    
    if args.data_root:
        config['data_root'] = args.data_root
        updated = True
    if args.epochs:
        config['num_epochs'] = args.epochs
        updated = True
    if args.batch_size:
        config['batch_size'] = args.batch_size
        updated = True
    if args.lr:
        config['learning_rate'] = args.lr
        updated = True
    if args.resume:
        config['resume_from'] = args.resume
        updated = True
    if args.save_dir:
        config['save_dir'] = args.save_dir
        updated = True
    if args.device:
        config['device'] = args.device
        updated = True
    if args.n_classes:
        config['n_classes'] = args.n_classes
        updated = True
    if args.loss_type:
        config['loss_type'] = args.loss_type
        updated = True
    if args.no_viz:
        config['enable_visualization'] = False
        updated = True
    if args.plot_interval:
        config['plot_interval'] = args.plot_interval
        updated = True
    if args.no_augmentation:
        config['use_augmentation'] = False
        updated = True
    if args.augmentation_type:
        config['augmentation_type'] = args.augmentation_type
        updated = True
    
    # Update Warmup parameters
    if args.momentum is not None:
        config['momentum'] = args.momentum
        updated = True
    if args.warmup_epochs is not None:
        config['warmup_epochs'] = args.warmup_epochs
        updated = True
    if args.warmup_momentum is not None:
        config['warmup_momentum'] = args.warmup_momentum
        updated = True
    if args.warmup_bias_lr is not None:
        config['warmup_bias_lr'] = args.warmup_bias_lr
        updated = True
    if args.scheduler is not None:
        config['scheduler'] = args.scheduler
        updated = True
    
    if updated:
        print("📝 Configuration overridden with command line arguments")
    
    return config

if __name__ == '__main__':
    args = parse_args()
    config = get_config()
    config = update_config_from_args(config, args)
    
    globals()['get_config'] = lambda: config
    main()