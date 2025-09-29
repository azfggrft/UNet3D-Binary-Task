#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3D UNet Integrated Visualization Training Script
Includes automatic visualization, early stopping mechanism, detailed monitoring
"""

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
import os


# Import from any file
from src.network_architecture.net_module import *
from src.network_architecture.unet3d import UNet3D
from .dataload import MedicalImageDataset, create_data_loaders
from src.loss_architecture.loss import DiceLoss, CombinedLoss
from src.loss_architecture.calculate_dice import calculate_dice_score, calculate_metrics

# ==================== Visualization Module ====================
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches

# Set font and visualization style
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")

class UNet3DVisualizer:
    """3D UNet Training Process Visualization Class"""
    
    def __init__(self, save_dir='./visualizations'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.colors = {
            'train': '#2E86AB',
            'val': '#A23B72', 
            'test': '#F18F01',
            'loss': '#C73E1D',
            'dice': '#3F8F3F',
            'lr': '#8B4513'
        }
        
        self.figsize_single = (10, 6)
        self.figsize_large = (15, 10)
    
    def plot_training_curves(self, history, title="Training curve", save_name="training_curves.png"):
        """Plot training and validation loss and Dice score curves"""
        fig, axes = plt.subplots(2, 2, figsize=self.figsize_large)
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Training and validation loss
        ax1 = axes[0, 0]
        ax1.plot(epochs, history['train_loss'], 
                color=self.colors['train'], linewidth=2, label='Training loss')
        if 'val_loss' in history and history['val_loss']:
            ax1.plot(epochs, history['val_loss'], 
                    color=self.colors['val'], linewidth=2, label='Validation loss')
        ax1.set_title('Loss curve', fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss value')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Training and validation Dice score
        ax2 = axes[0, 1]
        ax2.plot(epochs, history['train_dice'], 
                color=self.colors['train'], linewidth=2, label='Training Dice')
        if 'val_dice' in history and history['val_dice']:
            ax2.plot(epochs, history['val_dice'], 
                    color=self.colors['val'], linewidth=2, label='Validation Dice')
        ax2.set_title('Dice score curve', fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Dice score')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Learning rate change
        ax3 = axes[1, 0]
        if 'learning_rate' in history and history['learning_rate']:
            ax3.plot(epochs, history['learning_rate'], 
                    color=self.colors['lr'], linewidth=2, label='Learning rate')
            ax3.set_title('Learning rate change', fontweight='bold')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Learning rate')
            ax3.set_yscale('log')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'No learning rate data', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Learning rate change', fontweight='bold')
        
        # Training vs validation Dice scatter plot
        ax4 = axes[1, 1]
        if 'val_dice' in history and history['val_dice']:
            ax4.scatter(history['train_dice'], history['val_dice'], 
                       alpha=0.7, c=range(len(history['train_dice'])), 
                       cmap='viridis', s=50)
            
            min_val = min(min(history['train_dice']), min(history['val_dice']))
            max_val = max(max(history['train_dice']), max(history['val_dice']))
            ax4.plot([min_val, max_val], [min_val, max_val], 
                    'r--', alpha=0.8, linewidth=2, label='Perfect match line')
            
            ax4.set_title('Training vs Validation Dice scatter plot', fontweight='bold')
            ax4.set_xlabel('Training Dice')
            ax4.set_ylabel('Validation Dice')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            cbar = plt.colorbar(ax4.collections[0], ax=ax4)
            cbar.set_label('Epoch')
        else:
            ax4.text(0.5, 0.5, 'No validation data', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Training vs. Validation Dice scatter plot', fontweight='bold')
        
        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Training curves saved: {save_path}")
        plt.close()  # Close figure to save memory
    
    def plot_3d_predictions(self, images, masks, predictions, slice_indices=None, 
                           save_name="3d_predictions.png", max_samples=4):
        """Visualize 3D prediction results"""
        images = images.detach().cpu().numpy()
        masks = masks.detach().cpu().numpy()
        
        if predictions.dim() == 5:
            predictions = torch.softmax(predictions, dim=1)
            predictions = torch.argmax(predictions, dim=1)
        predictions = predictions.detach().cpu().numpy()
        
        batch_size = min(images.shape[0], max_samples)
        
        if slice_indices is None:
            depth = images.shape[2]
            slice_indices = [depth // 4, depth // 2, 3 * depth // 4]
        
        fig, axes = plt.subplots(batch_size, len(slice_indices) * 3, 
                               figsize=(len(slice_indices) * 9, batch_size * 3))
        
        if batch_size == 1:
            axes = axes.reshape(1, -1)
        
        for b in range(batch_size):
            for s_idx, slice_idx in enumerate(slice_indices):
                if slice_idx >= images.shape[2]:
                    slice_idx = images.shape[2] // 2
                
                col_base = s_idx * 3
                
                ax_img = axes[b, col_base]
                ax_img.imshow(images[b, 0, slice_idx], cmap='gray')
                ax_img.set_title(f'Sample{b+1} - Slice{slice_idx}\nOriginal image')
                ax_img.axis('off')
                
                ax_mask = axes[b, col_base + 1]
                ax_mask.imshow(masks[b, slice_idx], cmap='jet', alpha=0.7)
                ax_mask.set_title('Ground truth label')
                ax_mask.axis('off')
                
                ax_pred = axes[b, col_base + 2]
                ax_pred.imshow(predictions[b, slice_idx], cmap='jet', alpha=0.7)
                ax_pred.set_title('Predicted result')
                ax_pred.axis('off')
        
        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 3D prediction results saved: {save_path}")
        plt.close()
    
    def create_training_dashboard(self, history, save_name="training_dashboard.png"):
        """Create training dashboard"""
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1], width_ratios=[1, 1, 1, 1])
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Loss curve
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(epochs, history['train_loss'], label='train', color=self.colors['train'])
        if 'val_loss' in history and history['val_loss']:
            ax1.plot(epochs, history['val_loss'], label='validation', color=self.colors['val'])
        ax1.set_title('Loss curve', fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss value')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Dice score curve
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(epochs, history['train_dice'], label='train', color=self.colors['train'])
        if 'val_dice' in history and history['val_dice']:
            ax2.plot(epochs, history['val_dice'], label='validation', color=self.colors['val'])
        ax2.set_title('Dice score curve', fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Dice score')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Learning rate change
        ax3 = fig.add_subplot(gs[0, 2])
        if 'learning_rate' in history and history['learning_rate']:
            ax3.plot(epochs, history['learning_rate'], color=self.colors['lr'])
            ax3.set_yscale('log')
        ax3.set_title('Learning rate change', fontweight='bold')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning rate')
        ax3.grid(True, alpha=0.3)
        
        # Statistical information
        ax4 = fig.add_subplot(gs[0, 3])
        ax4.axis('off')
        
        best_train_dice = max(history['train_dice']) if history['train_dice'] else 0
        best_val_dice = max(history['val_dice']) if 'val_dice' in history and history['val_dice'] else 0
        final_train_loss = history['train_loss'][-1] if history['train_loss'] else 0
        final_val_loss = history['val_loss'][-1] if 'val_loss' in history and history['val_loss'] else 0
        
        stats_text = f"""
        Training statistics summary
        
        Best performance:
        • Best training Dice: {best_train_dice:.4f}
        • Best validation Dice: {best_val_dice:.4f}
        
        Final performance:
        • Final training loss: {final_train_loss:.4f}
        • Final validation loss: {final_val_loss:.4f}
        
        Number of training epochs: {len(epochs)}
        """
        ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, 
                fontsize=11, verticalalignment='top', fontweight='bold')
        
        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Training dashboard saved: {save_path}")
        plt.close()