import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import argparse
from pathlib import Path
import random
import numpy as np
import time
import json
import warnings
warnings.filterwarnings('ignore')

# Progress bar library
try:
    from tqdm import tqdm, trange
    TQDM_AVAILABLE = True
except ImportError:
    print("It is recommended to install tqdm for a better progress bar experience: pip install tqdm")
    TQDM_AVAILABLE = False

from src.network_architecture.net_module import *
from src.network_architecture.unet3d import UNet3D
from src.loss_architecture.loss import DiceLoss, CombinedLoss
from src.loss_architecture.calculate_dice import calculate_dice_score, calculate_metrics
from src.data_processing_and_data_enhancement.visualizer import UNet3DVisualizer


class EnhancedUNet3DTrainer:
    """Enhanced 3D UNet training manager with PyTorch 2.6+ loading fix"""

    def __init__(self, model, data_loaders, optimizer, criterion, device,
                 save_dir='./checkpoints', log_interval=10, save_interval=10,
                 scheduler=None, visualize=True, plot_interval=10,
                 use_progress_bar=True, training_config=None):  # Added parameter

        self.model = model.to(device) if model is not None else None
        self.data_loaders = data_loaders
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device if device is not None else torch.device('cpu')
        self.scheduler = scheduler
        self.save_dir = Path(save_dir) if save_dir is not None else Path('./checkpoints')
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.visualize = visualize
        self.plot_interval = plot_interval
        self.use_progress_bar = use_progress_bar and TQDM_AVAILABLE
        self.training_start_time = None
        self.training_end_time = None
        self.total_training_time = 0

        # Save training configuration (added)
        self.training_config = training_config if training_config is not None else {}

        # Compute model complexity info
        self.model_info = self._calculate_model_complexity()

        # Initialize visualizer
        if self.visualize:
            viz_dir = self.save_dir / 'visualizations'
            self.visualizer = UNet3DVisualizer(save_dir=viz_dir)
            print(f"Visualization enabled, images will be saved to: {viz_dir}")

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_dice': [],
            'val_dice': [],
            'learning_rate': [],
            'epoch_time': [],
            'gpu_memory': []
        }

        # Best model tracking
        self.best_val_dice = 0.0
        self.best_epoch = 0

        self.start_time = None
        self.epoch_start_time = None

        # Progress bar references
        self.epoch_pbar = None
        self.train_pbar = None
        self.val_pbar = None

    def _calculate_model_complexity(self):
        """Compute model complexity information"""
        try:
            # Get number of model parameters
            if hasattr(self.model, 'get_model_size'):
                total_params, trainable_params = self.model.get_model_size()
            else:
                total_params = sum(p.numel() for p in self.model.parameters())
                trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

            # Try to compute FLOPs (optional)
            try:
                from thop import profile, clever_format

                # Create sample input
                if hasattr(self.model, 'n_channels'):
                    n_channels = self.model.n_channels
                else:
                    n_channels = 1

                target_size = self.training_config.get('target_size', (64, 64, 64))
                sample_input = torch.randn(1, n_channels, *target_size).to(self.device)

                # Compute FLOPs
                flops, params = profile(self.model, inputs=(sample_input,), verbose=False)
                flops = flops / 2 / 1e9
                flops_str = f"{flops:.3f} GFLOPs"

                # Test inference time
                self.model.eval()
                inference_times = []

                with torch.no_grad():
                    for _ in range(3):  # warm-up
                        _ = self.model(sample_input)

                    for _ in range(10):
                        start_time = time.time()
                        _ = self.model(sample_input)
                        if self.device.type == 'cuda':
                            torch.cuda.synchronize()
                        end_time = time.time()
                        inference_times.append(end_time - start_time)

                avg_inference_time = np.mean(inference_times)

            except Exception as e:
                print(f"Error computing FLOPs: {e}")
                flops_str = "Unavailable"
                avg_inference_time = 0

            return {
                'total_params': total_params,
                'trainable_params': trainable_params,
                'model_size_mb': total_params * 4 / 1024 / 1024,  # FP32
                'flops': flops_str,
                'avg_inference_time': avg_inference_time
            }

        except Exception as e:
            print(f"Error computing model complexity: {e}")
            return {
                'total_params': 'Unknown',
                'trainable_params': 'Unknown',
                'model_size_mb': 0,
                'flops': 'Unknown',
                'avg_inference_time': 0
            }

    def safe_torch_load(self, path):
        """Safe torch.load function compatible with PyTorch 2.6+"""
        try:
            return torch.load(path, map_location=self.device, weights_only=False)
        except Exception as e1:
            try:
                print(f"Failed with weights_only=False, trying weights_only=True")
                return torch.load(path, map_location=self.device, weights_only=True)
            except Exception as e2:
                print(f"Setting safe globals and retrying load...")
                torch.serialization.add_safe_globals([
                    'numpy._core.multiarray.scalar',
                    'numpy.core.multiarray.scalar',
                    'numpy.dtype',
                    'numpy.ndarray',
                    'collections.OrderedDict'
                ])
                return torch.load(path, map_location=self.device, weights_only=True)

    def log_gpu_memory(self):
        """Log GPU memory usage"""
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(self.device) / 1024**3
            memory_cached = torch.cuda.memory_reserved(self.device) / 1024**3
            self.history['gpu_memory'].append({
                'allocated': memory_allocated,
                'cached': memory_cached
            })
        else:
            self.history['gpu_memory'].append({'allocated': 0, 'cached': 0})

    def _format_dice_score(self, dice_score):
        """
        Safely format Dice score, handling lists and single values

        Args:
            dice_score: single value or list/tuple

        Returns:
            str: formatted string
        """
        if isinstance(dice_score, (list, tuple)):
            if len(dice_score) == 1:
                return f'{dice_score[0]:.4f}'
            else:
                mean_dice = np.mean(dice_score)
                return f'{mean_dice:.4f}'
        else:
            return f'{dice_score:.4f}'

    def train_one_epoch(self, epoch):
        """Train one epoch - fixes learning rate display"""
        self.epoch_start_time = time.time()
        self.model.train()
        train_loss = 0.0
        train_dice_scores = []

        num_batches = len(self.data_loaders['train'])

        # Check warmup stage
        has_warmup = hasattr(self, 'warmup_scheduler') and self.warmup_scheduler is not None
        warmup_epochs = getattr(self, 'warmup_epochs', 0) if has_warmup else 0
        is_warmup_stage = has_warmup and epoch < warmup_epochs

        # Create training progress bar
        if self.use_progress_bar:
            desc_prefix = "🔥 Warmup" if is_warmup_stage else "🚀 Train"
            self.train_pbar = tqdm(
                enumerate(self.data_loaders['train']),
                total=num_batches,
                desc=f"{desc_prefix} E{epoch+1:3d}",
                leave=False,
                ncols=100,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}'
            )
        else:
            stage_info = "Warmup training" if is_warmup_stage else "Normal training"
            print(f"\nStarting {stage_info} Epoch {epoch+1}")

        data_iter = self.train_pbar if self.use_progress_bar else enumerate(self.data_loaders['train'])

        for batch_idx, batch in data_iter:
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)

            # Ensure masks are binary
            masks[masks > 0] = 1

            self.optimizer.zero_grad()
            outputs = self.model(images)

            loss = self.criterion(outputs, masks)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            train_loss += loss.item()

            with torch.no_grad():
                dice_score = calculate_dice_score(
                    outputs, masks, num_classes=self.model.n_classes
                )
                train_dice_scores.append(dice_score)

            # Update progress bar - show actual learning rate
            if self.use_progress_bar:
                current_lr = self.optimizer.param_groups[0]['lr']

                # Get momentum info
                momentum_info = ""
                if 'momentum' in self.optimizer.param_groups[0]:
                    momentum = self.optimizer.param_groups[0]['momentum']
                    momentum_info = f", M:{momentum:.2f}"
                elif 'betas' in self.optimizer.param_groups[0]:
                    beta1 = self.optimizer.param_groups[0]['betas'][0]
                    momentum_info = f", β1:{beta1:.2f}"

                postfix_data = {
                    'Loss': f'{loss.item():.4f}',
                    'Dice': self._format_dice_score(dice_score),
                    'LR': f'{current_lr:.2e}'
                }

                if is_warmup_stage:
                    postfix_data['Stage'] = f'W{epoch+1}/{warmup_epochs}'

                self.train_pbar.set_postfix(postfix_data)

            elif batch_idx % 5 == 0:
                progress = 100.0 * batch_idx / num_batches
                current_lr = self.optimizer.param_groups[0]['lr']
                dice_str = self._format_dice_score(dice_score)

                stage_prefix = "[WARMUP]" if is_warmup_stage else "[TRAIN]"
                print(f'\r  {stage_prefix} Batch [{batch_idx:3d}/{num_batches}] {progress:5.1f}% | '
                      f'Loss: {loss.item():.4f} | Dice: {dice_str} | LR: {current_lr:.6f}', end='')

        if self.use_progress_bar and self.train_pbar:
            self.train_pbar.close()

        avg_train_loss = train_loss / num_batches

        if train_dice_scores:
            if isinstance(train_dice_scores[0], (list, tuple)):
                avg_train_dice = np.mean([score[0] if len(score) > 0 else 0.0 for score in train_dice_scores])
            else:
                avg_train_dice = np.mean(train_dice_scores)
        else:
            avg_train_dice = 0.0

        epoch_time = time.time() - self.epoch_start_time
        self.history['epoch_time'].append(epoch_time)
        self.log_gpu_memory()

        if not self.use_progress_bar:
            stage_info = "Warmup training" if is_warmup_stage else "Training"
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f'\n  {stage_info} completed | Avg Loss: {avg_train_loss:.4f} | Avg Dice: {avg_train_dice:.4f} | '
                  f'Current LR: {current_lr:.6f} | Time: {epoch_time:.1f}s')

        return avg_train_loss, avg_train_dice

    
    def validate_one_epoch(self, epoch):
        """Validate a single epoch"""
        if 'val' not in self.data_loaders:
            return 0.0, 0.0

        self.model.eval()
        val_loss = 0.0
        val_dice_scores = []

        num_batches = len(self.data_loaders['val'])

        # Create validation progress bar
        if self.use_progress_bar:
            self.val_pbar = tqdm(
                enumerate(self.data_loaders['val']),
                total=num_batches,
                desc=f"Epoch {epoch+1:3d} Validation",
                leave=False,
                ncols=100,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}'
            )
        else:
            print("Starting validation...")

        data_iter = self.val_pbar if self.use_progress_bar else enumerate(self.data_loaders['val'])

        with torch.no_grad():
            for batch_idx, batch in data_iter:
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                masks[masks > 0] = 1  # Ensure binary mask

                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                val_loss += loss.item()

                dice_score = calculate_dice_score(outputs, masks, num_classes=self.model.n_classes)
                val_dice_scores.append(dice_score)

                # Update progress bar
                if self.use_progress_bar:
                    self.val_pbar.set_postfix({
                        'Loss': f'{loss.item():.4f}',
                        'Dice': self._format_dice_score(dice_score)
                    })
                elif batch_idx % 3 == 0:
                    progress = 100.0 * batch_idx / num_batches
                    dice_str = self._format_dice_score(dice_score)
                    print(f'\r  Validation progress {progress:5.1f}% | Current Dice: {dice_str}', end='')

        # Close validation progress bar
        if self.use_progress_bar and self.val_pbar:
            self.val_pbar.close()

        avg_val_loss = val_loss / num_batches

        # Compute average Dice score
        if val_dice_scores:
            if isinstance(val_dice_scores[0], (list, tuple)):
                avg_val_dice = np.mean([score[0] if len(score) > 0 else 0.0 for score in val_dice_scores])
            else:
                avg_val_dice = np.mean(val_dice_scores)
        else:
            avg_val_dice = 0.0

        if not self.use_progress_bar:
            print(f'\n  Validation completed | Avg Loss: {avg_val_loss:.4f} | Avg Dice: {avg_val_dice:.4f}')

        return avg_val_loss, avg_val_dice


    def visualize_epoch_results(self, epoch, sample_predictions=None):
        """Visualize results of the current epoch"""
        if not self.visualize:
            return

        try:
            if epoch % self.plot_interval == 0:
                if self.use_progress_bar:
                    tqdm.write("Updating training curves...")
                else:
                    print("Updating training curves...")

                if hasattr(self.visualizer, 'plot_training_curves'):
                    self.visualizer.plot_training_curves(
                        self.history,
                        title=f"Training curve (up to Epoch {epoch+1})",
                        save_name=f"training_curves_epoch_{epoch+1}.png"
                    )
                else:
                    print("Visualizer does not have method 'plot_training_curves'")

            if sample_predictions is not None:
                images, masks, predictions = sample_predictions
                if hasattr(self.visualizer, 'plot_3d_predictions'):
                    self.visualizer.plot_3d_predictions(
                        images, masks, predictions,
                        save_name=f"predictions_epoch_{epoch+1}.png",
                        max_samples=4
                    )
                else:
                    print("Visualizer does not have method 'plot_3d_predictions'")

        except Exception as e:
            if self.use_progress_bar:
                tqdm.write(f"Visualization error: {e}")
            else:
                print(f"Visualization error: {e}")
            # Do not stop training

    
    def save_checkpoint(self, epoch, is_best=False):
        """Save a training checkpoint using the actual training configuration instead of model defaults."""
        # Extract model configuration from training_config if available
        if self.training_config:
            model_config = {
                'n_channels': self.training_config.get('n_channels', getattr(self.model, 'n_channels', 1)),
                'n_classes': self.training_config.get('n_classes', getattr(self.model, 'n_classes', 2)),
                'base_channels': self.training_config.get('base_channels', getattr(self.model, 'base_channels', 64)),
                'num_groups': self.training_config.get('num_groups', getattr(self.model, 'num_groups', 8)),
                'bilinear': self.training_config.get('bilinear', getattr(self.model, 'bilinear', False))
            }
        else:
            # Fallback: dynamically extract from model
            model_config = {
                'n_channels': getattr(self.model, 'n_channels', 1),
                'n_classes': getattr(self.model, 'n_classes', 2),
                'base_channels': getattr(self.model, 'base_channels', 64),
                'num_groups': getattr(self.model, 'num_groups', 8),
                'bilinear': getattr(self.model, 'bilinear', False)
            }

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.history['train_loss'],
            'val_losses': self.history['val_loss'],
            'train_dice': self.history['train_dice'],
            'val_dice': self.history['val_dice'],
            'learning_rate': self.history['learning_rate'],
            'best_val_dice': self.best_val_dice,
            'total_training_time': self.total_training_time,
            'model_config': model_config,
            'training_config': self.training_config,
            'model_info': self.model_info
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        latest_path = self.save_dir / 'latest_checkpoint.pth'
        torch.save(checkpoint, latest_path)

        if is_best:
            best_path = self.save_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"Best model saved: {best_path}")

        if (epoch + 1) % (self.save_interval * 2) == 0:
            print(f"Checkpoint saved to: {latest_path}")


    def load_checkpoint(self, checkpoint_path):
        """Load a model checkpoint (PyTorch 2.6 compatible)."""
        try:
            checkpoint = self.safe_torch_load(checkpoint_path)
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            raise

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_dice = checkpoint.get('best_val_dice', 0.0)
        self.history = checkpoint.get('history', self.history)

        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        start_epoch = checkpoint['epoch'] + 1
        print(f"Checkpoint loaded, resuming from epoch {start_epoch}")
        print(f"Best historical Dice: {self.best_val_dice:.4f}")
        return start_epoch


    def get_sample_predictions(self):
        """Get sample predictions for visualization from the validation set."""
        if 'val' not in self.data_loaders:
            return None

        self.model.eval()
        with torch.no_grad():
            for batch in self.data_loaders['val']:
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                masks[masks > 0] = 1
                outputs = self.model(images)
                return (images[:4], masks[:4], outputs[:4])
        return None


    def train(self, num_epochs, resume_from=None, early_stopping_patience=None):
        """Train the model with proper warmup learning rate display."""
        print(f"Starting training for {num_epochs} epochs")

        self.training_start_time = time.time()
        start_epoch = 0

        has_warmup = hasattr(self, 'warmup_scheduler') and self.warmup_scheduler is not None
        warmup_epochs = getattr(self, 'warmup_epochs', 0) if has_warmup else 0

        if has_warmup:
            print(f"🔥 Warmup enabled: first {warmup_epochs} epochs will use warmup scheduling")

        # Resume from checkpoint if provided
        if resume_from and Path(resume_from).exists():
            checkpoint = self.safe_torch_load(resume_from)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1

            self.history['train_loss'] = checkpoint.get('train_losses', [])
            self.history['val_loss'] = checkpoint.get('val_losses', [])
            self.history['train_dice'] = checkpoint.get('train_dice', [])
            self.history['val_dice'] = checkpoint.get('val_dice', [])

            if 'learning_rate' in checkpoint:
                self.history['learning_rate'] = checkpoint['learning_rate']
            else:
                current_lr = self.optimizer.param_groups[0]['lr']
                self.history['learning_rate'] = [current_lr] * len(self.history['train_loss'])

            self.best_val_dice = checkpoint.get('best_val_dice', 0)
            self.total_training_time = checkpoint.get('total_training_time', 0)

            print(f"Resuming from epoch {start_epoch}")
            print(f"Total trained time: {self.total_training_time/3600:.2f} hours")
            print(f"Current learning rate: {self.optimizer.param_groups[0]['lr']:.6f}")

            if has_warmup and start_epoch < warmup_epochs:
                self.warmup_scheduler.current_epoch = start_epoch
                print(f"🔥 Restored warmup state: epoch {start_epoch}/{warmup_epochs}")

        early_stopping_counter = 0

        try:
            for epoch in range(start_epoch, num_epochs):
                epoch_start_time = time.time()

                # Warmup step
                if has_warmup and epoch < warmup_epochs:
                    self.warmup_scheduler.step(epoch)
                    warmup_stage = True
                    current_lrs = self.warmup_scheduler.get_lr()
                    current_momenta = self.warmup_scheduler.get_momentum()
                    msg = f"🔥 Warmup stage [{epoch+1}/{warmup_epochs}] - LR: {current_lrs[0]:.6f}, Momentum: {current_momenta[0]:.3f}"
                    print(msg) if not self.use_progress_bar else tqdm.write(msg)
                else:
                    warmup_stage = False

                train_loss, train_dice = self.train_one_epoch(epoch)
                val_loss, val_dice = self.validate_one_epoch(epoch)

                self.history['train_loss'].append(train_loss)
                self.history['val_loss'].append(val_loss)
                self.history['train_dice'].append(train_dice)
                self.history['val_dice'].append(val_dice)
                current_lr = self.optimizer.param_groups[0]['lr']
                self.history['learning_rate'].append(current_lr)

                epoch_time = time.time() - epoch_start_time
                self.total_training_time += epoch_time

                # Scheduler step
                if not warmup_stage and self.scheduler:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_dice)
                    else:
                        self.scheduler.step()

                # Check best model
                is_best = val_dice > self.best_val_dice
                if is_best:
                    self.best_val_dice = val_dice
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1

                # Logging
                if (epoch + 1) % self.log_interval == 0 or is_best:
                    momentum_info = ""
                    if 'momentum' in self.optimizer.param_groups[0]:
                        momentum_info = f", Momentum: {self.optimizer.param_groups[0]['momentum']:.3f}"
                    elif 'betas' in self.optimizer.param_groups[0]:
                        momentum_info = f", Beta1: {self.optimizer.param_groups[0]['betas'][0]:.3f}"
                    stage_info = "🔥 [WARMUP]" if warmup_stage else "🚀 [NORMAL]"
                    log_msg = (f"{stage_info} Epoch [{epoch+1}/{num_epochs}] "
                            f"Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f} | "
                            f"Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f} | "
                            f"LR: {current_lr:.6f}{momentum_info} | Time: {epoch_time:.1f}s")
                    if is_best:
                        log_msg += " ⭐ [NEW BEST]"
                    print(log_msg) if not self.use_progress_bar else tqdm.write(log_msg)

                # Save checkpoint
                if (epoch + 1) % self.save_interval == 0 or is_best:
                    self.save_checkpoint(epoch, is_best)

                # Visualization
                if self.visualize and (epoch + 1) % self.plot_interval == 0:
                    try:
                        self.visualizer.plot_training_curves(self.history, title=f"Training curve (up to Epoch {epoch+1})", save_name=f"training_curves_epoch_{epoch+1:03d}.png")
                        sample_predictions = self.get_sample_predictions()
                        if sample_predictions and hasattr(self.visualizer, 'plot_3d_predictions'):
                            images, masks, predictions = sample_predictions
                            self.visualizer.plot_3d_predictions(images, masks, predictions, save_name=f"predictions_epoch_{epoch+1:03d}.png", max_samples=3)
                            msg = f"Generated predictions for epoch {epoch+1}"
                            print(msg) if not self.use_progress_bar else tqdm.write(msg)
                    except Exception as e:
                        print(f"Visualization error: {e}")

                # Early stopping
                if early_stopping_patience and early_stopping_counter >= early_stopping_patience:
                    print(f"Early stopping triggered! No improvement for {early_stopping_patience} epochs")
                    break

            self.training_end_time = time.time()
            print("Training completed!")
            print(f"Total training time: {self.total_training_time/3600:.2f} hours")
            print(f"Best validation Dice: {self.best_val_dice:.4f}")

            # Final visualization
            if self.visualize:
                try:
                    if hasattr(self.visualizer, 'create_final_dashboard'):
                        self.visualizer.create_final_dashboard(self.history)
                    elif hasattr(self.visualizer, 'create_training_dashboard'):
                        self.visualizer.create_training_dashboard(self.history, "final_training_dashboard.png")
                    else:
                        self.visualizer.plot_training_curves(self.history, len(self.history['train_loss']))
                except Exception as e:
                    print(f"Final visualization failed: {e}, skipping visualization step")

        except KeyboardInterrupt:
            print("Training interrupted")
            self.training_end_time = time.time()

        return self.history


    
    def test(self, checkpoint_path=None, save_results=True):
        """Test the model, generate visual results, and save a detailed test report."""
        if 'test' not in self.data_loaders:
            print("No test dataset available")
            return None
        
        if checkpoint_path:
            try:
                checkpoint = self.safe_torch_load(checkpoint_path)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Loaded test model: {checkpoint_path}")
            except Exception as e:
                print(f"Failed to load test model: {e}")
                return None
        
        self.model.eval()
        test_dice_scores = []
        test_losses = []
        all_predictions = []
        test_details = []  # Record detailed info
        
        num_batches = len(self.data_loaders['test'])
        
        # Create test progress bar
        if self.use_progress_bar:
            test_pbar = tqdm(
                enumerate(self.data_loaders['test']),
                total=num_batches,
                desc="Testing Model",
                ncols=100,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}'
            )
        else:
            print("Starting testing...")
        
        data_iter = test_pbar if self.use_progress_bar else enumerate(self.data_loaders['test'])
        
        with torch.no_grad():
            for batch_idx, batch in data_iter:
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                masks[masks > 0] = 1
                
                outputs = self.model(images)

                # Compute batch loss (for progress display)
                batch_loss = self.criterion(outputs, masks)
                test_losses.append(batch_loss.item())

                # Compute batch Dice score (for progress display)
                batch_metrics = calculate_metrics(outputs, masks, self.model.n_classes)
                batch_dice_score = batch_metrics['mean_dice']
                test_dice_scores.append(batch_dice_score)

                # Compute individual loss and Dice per sample
                batch_size = images.size(0)
                individual_losses = []
                individual_dice_scores = []

                for sample_idx in range(batch_size):
                    single_image = images[sample_idx:sample_idx+1]
                    single_mask = masks[sample_idx:sample_idx+1]
                    single_output = outputs[sample_idx:sample_idx+1]
                    
                    single_loss = self.criterion(single_output, single_mask)
                    individual_losses.append(single_loss.item())
                    
                    single_metrics = calculate_metrics(single_output, single_mask, self.model.n_classes)
                    single_dice = single_metrics['mean_dice']
                    individual_dice_scores.append(single_dice)

                # Record detailed info (per file)
                if 'image_path' in batch:
                    for i, img_path in enumerate(batch['image_path']):
                        if i < len(individual_losses):
                            test_details.append({
                                'file': Path(img_path).name,
                                'loss': individual_losses[i],
                                'dice': individual_dice_scores[i],
                                'batch_idx': batch_idx,
                                'sample_idx': i
                            })
                else:
                    for i in range(len(individual_losses)):
                        test_details.append({
                            'file': f'batch_{batch_idx:03d}_sample_{i:02d}',
                            'loss': individual_losses[i],
                            'dice': individual_dice_scores[i],
                            'batch_idx': batch_idx,
                            'sample_idx': i
                        })
                
                # Collect predictions for visualization
                if batch_idx < 3 and self.visualize:
                    all_predictions.append((images.cpu(), masks.cpu(), outputs.cpu()))
                
                # Update progress
                if self.use_progress_bar:
                    test_pbar.set_postfix({
                        'Loss': f'{batch_loss.item():.4f}',
                        'Dice': f'{batch_dice_score:.4f}'
                    })
                elif batch_idx % 5 == 0:
                    print(f"Testing progress: {batch_idx+1}/{num_batches}")
        
        # Close progress bar
        if self.use_progress_bar and 'test_pbar' in locals():
            test_pbar.close()
        
        # ============== Key fix: Compute real statistics per sample ==============
        if test_details:
            individual_losses = [detail['loss'] for detail in test_details]
            individual_dice_scores = [detail['dice'] for detail in test_details]
            
            avg_test_loss = np.mean(individual_losses)
            avg_test_dice = np.mean(individual_dice_scores)
            
            loss_std = np.std(individual_losses)
            dice_std = np.std(individual_dice_scores)
            min_loss = np.min(individual_losses)
            max_loss = np.max(individual_losses)
            min_dice = np.min(individual_dice_scores)
            max_dice = np.max(individual_dice_scores)
            
            final_losses = individual_losses
            final_dice_scores = individual_dice_scores
            sample_count = len(test_details)
            unit_description = "files"
        else:
            avg_test_loss = np.mean(test_losses)
            avg_test_dice = np.mean(test_dice_scores)
            
            loss_std = np.std(test_losses)
            dice_std = np.std(test_dice_scores)
            min_loss = np.min(test_losses)
            max_loss = np.max(test_losses)
            min_dice = np.min(test_dice_scores)
            max_dice = np.max(test_dice_scores)
            
            final_losses = test_losses
            final_dice_scores = test_dice_scores
            sample_count = len(test_losses)
            unit_description = "batches"
        
        result_msg = f"""
        Test Results:
        Number of test samples: {sample_count} {unit_description}
        Average Loss: {avg_test_loss:.4f}
        Average Dice Score: {avg_test_dice:.4f} ({avg_test_dice * 100:.2f}%)
        Loss Range: {min_loss:.4f} ~ {max_loss:.4f} (Std: {loss_std:.4f})
        Dice Range: {min_dice:.4f} ~ {max_dice:.4f} (Std: {dice_std:.4f})"""
        
        if self.use_progress_bar:
            tqdm.write(result_msg)
        else:
            print(result_msg)
        
        # Save test results to file
        if save_results:
            self._save_test_results(
                avg_test_loss, avg_test_dice,
                final_losses, final_dice_scores,
                test_details, checkpoint_path,
                loss_std, dice_std, min_loss, max_loss, min_dice, max_dice,
                sample_count, unit_description
            )
        
        # Generate test visualization
        if self.visualize and all_predictions:
            viz_msg = "Generating test visualizations..."
            if self.use_progress_bar:
                tqdm.write(viz_msg)
            else:
                print(viz_msg)
            
            try:
                for i, (images, masks, predictions) in enumerate(all_predictions):
                    if hasattr(self.visualizer, 'plot_3d_predictions'):
                        self.visualizer.plot_3d_predictions(
                            images, masks, predictions,
                            save_name=f"test_results_batch_{i+1}.png"
                        )
                    else:
                        print("Visualizer has no plot_3d_predictions method, skipping visualization")
                        break
            except Exception as e:
                error_msg = f"Failed to visualize test results: {e}"
                if self.use_progress_bar:
                    tqdm.write(error_msg)
                else:
                    print(error_msg)
        
        return {
            'avg_loss': avg_test_loss,
            'avg_dice': avg_test_dice,
            'all_losses': final_losses,
            'all_dice_scores': final_dice_scores,
            'details': test_details,
            'stats': {
                'loss_std': loss_std,
                'dice_std': dice_std,
                'min_loss': min_loss,
                'max_loss': max_loss,
                'min_dice': min_dice,
                'max_dice': max_dice,
                'sample_count': sample_count,
                'unit': unit_description
            }
        }

    def _save_test_results(self, avg_loss, avg_dice, all_losses, all_dice, details, model_path,
                       loss_std, dice_std, min_loss, max_loss, min_dice, max_dice,
                       sample_count, unit_description):
        """Enhanced test result saving - uses actual training configuration."""
        import datetime
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Load additional info from checkpoint
        training_info = {}
        model_config = {}
        model_complexity = {}
        full_training_config = {}  # New: full training config
        
        if model_path and Path(model_path).exists():
            try:
                checkpoint = self.safe_torch_load(model_path)
                training_info = {
                    'total_epochs': checkpoint.get('epoch', 'Unknown') + 1,
                    'best_val_dice': checkpoint.get('best_val_dice', 'Unknown'),
                    'total_training_time': checkpoint.get('total_training_time', 0),
                    'final_train_loss': checkpoint.get('train_losses', [])[-1] if checkpoint.get('train_losses') else 'Unknown',
                    'final_val_loss': checkpoint.get('val_losses', [])[-1] if checkpoint.get('val_losses') else 'Unknown'
                }
                # Prefer actual config from checkpoint
                model_config = checkpoint.get('model_config', {})
                full_training_config = checkpoint.get('training_config', {})
                model_complexity = checkpoint.get('model_info', self.model_info)
            except:
                pass
        
        # If checkpoint does not have config, use current trainer config
        if not model_config and self.training_config:
            model_config = {
                'n_channels': self.training_config.get('n_channels', 1),
                'n_classes': self.training_config.get('n_classes', 2),
                'base_channels': self.training_config.get('base_channels', 32),
                'num_groups': self.training_config.get('num_groups', 8),
                'bilinear': self.training_config.get('bilinear', False)
            }
        
        # Save to TXT file
        txt_file = self.save_dir / f"test_results_{timestamp}.txt"
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("3D UNet Model Full Test Report\n")
            f.write("=" * 70 + "\n")
            f.write(f"Test Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model Path: {model_path or 'current training model'}\n")
            f.write("\n")
            
            # Model basic info (using actual config)
            f.write("=" * 25 + " Model Basic Info " + "=" * 25 + "\n")
            f.write(f"Input Channels (n_channels): {model_config.get('n_channels', 'Unknown')}\n")
            f.write(f"Output Classes (n_classes): {model_config.get('n_classes', 'Unknown')}\n")
            f.write(f"Base Channels (base_channels): {model_config.get('base_channels', 'Unknown')}\n")
            f.write(f"GroupNorm Groups (num_groups): {model_config.get('num_groups', 'Unknown')}\n")
            f.write(f"Bilinear Upsampling (bilinear): {model_config.get('bilinear', 'Unknown')}\n")
            f.write("\n")
            
            # If full training config exists, show key training parameters
            if full_training_config:
                f.write("=" * 24 + " Training Config " + "=" * 24 + "\n")
                f.write(f"Batch Size: {full_training_config.get('batch_size', 'Unknown')}\n")
                f.write(f"Learning Rate: {full_training_config.get('learning_rate', 'Unknown')}\n")
                f.write(f"Optimizer: {full_training_config.get('optimizer', 'Unknown')}\n")
                f.write(f"Loss Function: {full_training_config.get('loss_type', 'Unknown')}\n")
                f.write(f"Data Augmentation: {full_training_config.get('use_augmentation', 'Unknown')}\n")
                if full_training_config.get('use_augmentation'):
                    f.write(f"Augmentation Type: {full_training_config.get('augmentation_type', 'Unknown')}\n")
                f.write(f"Target Image Size: {full_training_config.get('target_size', 'Unknown')}\n")
                f.write("\n")
            
            # Model complexity info
            f.write("=" * 25 + " Model Complexity Info " + "=" * 24 + "\n")
            f.write(f"Total Parameters: {model_complexity.get('total_params', 'Unknown'):,}\n")
            f.write(f"Trainable Parameters: {model_complexity.get('trainable_params', 'Unknown'):,}\n")
            f.write(f"Model Size: {model_complexity.get('model_size_mb', 0):.2f} MB (FP32)\n")
            f.write(f"GLOPs: {model_complexity.get('flops', 'Unknown')}\n")
            f.write(f"Avg Inference Time: {model_complexity.get('avg_inference_time', 0)*1000:.2f} ms\n")
            f.write("\n")
            
            # Training info
            f.write("=" * 26 + " Training Info " + "=" * 26 + "\n")
            if training_info:
                f.write(f"Total Epochs: {training_info.get('total_epochs', 'Unknown')} epochs\n")
                total_time = training_info.get('total_training_time', 0)
                if total_time > 0:
                    hours = int(total_time // 3600)
                    minutes = int((total_time % 3600) // 60)
                    seconds = int(total_time % 60)
                    f.write(f"Total Training Time: {hours:02d}:{minutes:02d}:{seconds:02d} ({total_time/3600:.2f} hours)\n")
                    f.write(f"Avg Time per Epoch: {total_time/training_info.get('total_epochs', 1):.1f} sec\n")
                f.write(f"Final Train Loss: {training_info.get('final_train_loss', 'Unknown')}\n")
                f.write(f"Final Validation Loss: {training_info.get('final_val_loss', 'Unknown')}\n")
                f.write(f"Best Validation Dice: {training_info.get('best_val_dice', 'Unknown')}\n")
            else:
                f.write("Unable to load training info\n")
            f.write("\n")
            
            # Test results
            f.write("=" * 27 + " Test Results " + "=" * 27 + "\n")
            f.write(f"Number of Test Samples: {sample_count} {unit_description}\n")
            f.write(f"Average Test Loss: {avg_loss:.6f}\n")
            f.write(f"Average Test Dice: {avg_dice:.6f} ({avg_dice * 100:.2f}%)\n")
            f.write("\n")
            
            # Statistics
            f.write("=" * 27 + " Statistics " + "=" * 27 + "\n")
            f.write(f"Loss - Std: {loss_std:.6f}\n")
            f.write(f"Loss - Min: {min_loss:.6f}\n")
            f.write(f"Loss - Max: {max_loss:.6f}\n")
            f.write(f"Dice - Std: {dice_std:.6f}\n")
            f.write(f"Dice - Min: {min_dice:.6f}\n")
            f.write(f"Dice - Max: {max_dice:.6f}\n")
            f.write("\n")
            
            # Performance evaluation
            f.write("=" * 27 + " Performance Evaluation " + "=" * 27 + "\n")
            if avg_dice >= 0.90:
                performance = "Excellent"
                evaluation = "Model performs very well, suitable for real applications"
            elif avg_dice >= 0.80:
                performance = "Good"
                evaluation = "Model performs well, further optimization can be considered"
            elif avg_dice >= 0.75:
                performance = "Fair"
                evaluation = "Model performance is moderate; consider tuning hyperparameters or adding data"
            else:
                performance = "Needs Improvement"
                evaluation = "Model performance is poor; check data quality and model architecture"
                
            f.write(f"Overall Rating: {performance} (Dice: {avg_dice:.4f})\n")
            f.write(f"Evaluation Suggestion: {evaluation}\n")
            f.write("\n")
            
            # Detailed results (if not too many)
            if details and len(details) <= 30:
                f.write("=" * 25 + " Detailed Test Results " + "=" * 25 + "\n")
                f.write(f"{'File/Batch':<35} {'Loss':<12} {'Dice Score':<12}\n")
                f.write("-" * 65 + "\n")
                for detail in details:
                    f.write(f"{detail['file']:<35} {detail['loss']:<12.6f} {detail['dice']:<12.6f}\n")
            
            f.write("\n" + "=" * 70 + "\n")
            f.write("End of Report\n")
        
        print(f"Full test report saved to: {txt_file}")
        
        # Save in JSON format (including all data)
        json_file = self.save_dir / f"test_results_{timestamp}.json"
        results_data = {
            'timestamp': datetime.datetime.now().isoformat(),
            'model_path': str(model_path) if model_path else 'current_model',
            'model_config': model_config,
            'training_config': full_training_config,  # Full training config
            'model_complexity': model_complexity,
            'training_info': training_info,
            'test_summary': {
                'avg_loss': float(avg_loss),
                'avg_dice': float(avg_dice),
                'sample_count': sample_count,
                'unit': unit_description,
                'loss_std': float(loss_std),
                'dice_std': float(dice_std),
                'best_dice': float(max_dice),
                'worst_dice': float(min_dice),
                'min_loss': float(min_loss),
                'max_loss': float(max_loss)
            },
            'all_losses': [float(x) for x in all_losses],
            'all_dice_scores': [float(x) for x in all_dice],
            'details': details
        }
        
        import json
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, ensure_ascii=False, indent=2)
        
        print(f"Full JSON data saved to: {json_file}")
