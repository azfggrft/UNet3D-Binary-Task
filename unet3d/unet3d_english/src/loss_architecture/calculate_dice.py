import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset, DataLoader
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from .loss import DiceLoss, CombinedLoss

# ==================== Dice Score Calculation ====================
def calculate_dice_score(predictions, targets, num_classes, smooth=1e-6):
    """
    Compute Dice score - only for the foreground class (label 1)
    
    Args:
        predictions: [B, C, D, H, W] or [B, D, H, W] predicted results
        targets: [B, D, H, W] ground truth labels
        num_classes: number of classes
        smooth: smoothing term
        
    Returns:
        dice_score: Dice score for the foreground class (float)
    """
    # If predictions have a channel dimension, take argmax
    if predictions.dim() == 5:
        predictions = torch.argmax(predictions, dim=1)  # [B, D, H, W]
    
    # Only calculate for foreground class (label 1)
    class_idx = 1
    
    # Create binary masks - only for the foreground class
    pred_mask = (predictions == class_idx).float()
    target_mask = (targets == class_idx).float()
    
    # Compute intersection and union
    intersection = (pred_mask * target_mask).sum()
    union = pred_mask.sum() + target_mask.sum()
    
    # Compute Dice score
    if union == 0:
        dice = 1.0 if intersection == 0 else 0.0
    else:
        dice = (2. * intersection + smooth) / (union + smooth)
    
    return dice.item()


def calculate_metrics(predictions, targets, num_classes):
    """
    Compute multiple evaluation metrics - optimized for foreground class
    """
    if predictions.dim() == 5:
        predictions = torch.argmax(predictions, dim=1)
    
    # Overall accuracy
    accuracy = (predictions == targets).float().mean().item()
    
    # Dice score for foreground class
    dice_score = calculate_dice_score(predictions, targets, num_classes)
    
    return {
        'accuracy': accuracy,
        'mean_dice': dice_score,
        'foreground_dice': dice_score,
        'dice_per_class': [dice_score]  # maintain backward compatibility
    }
