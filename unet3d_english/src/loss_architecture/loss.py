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

# ==================== Dice Loss Implementation ====================
class DiceLoss(nn.Module):
    """
    Implementation of Dice Loss, suitable for multi-class segmentation
    """
    def __init__(self, smooth=1e-6, ignore_index=None, weight=None):
        """
        Args:
            smooth: smoothing term to avoid division by zero
            ignore_index: class index to ignore (e.g., background)
            weight: per-class weights
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.weight = weight
        
    def forward(self, predictions, targets):
        """
        Args:
            predictions: [B, C, D, H, W] model output logits
            targets: [B, D, H, W] or [B, C, D, H, W] ground truth labels
        """
        # Convert logits to probabilities
        predictions = F.softmax(predictions, dim=1)
        
        # If targets are [B, D, H, W], convert to one-hot
        if targets.dim() == 4:
            min_val = targets.min().item()
            max_val = targets.max().item()
            num_classes = predictions.size(1)
            
            if min_val < 0:
                print(f"Warning: Found negative label value {min_val}, setting it to 0")
                targets = torch.clamp(targets, min=0)
            
            if max_val >= num_classes:
                print(f"Warning: Label value {max_val} exceeds class range [0, {num_classes-1}], clipping it")
                targets = torch.clamp(targets, max=num_classes-1)
            
            # Convert to one-hot encoding
            targets = F.one_hot(targets.long(), num_classes=num_classes)
            targets = targets.permute(0, 4, 1, 2, 3).float()  # [B, C, D, H, W]
        
        # Compute Dice coefficient for each class
        dice_scores = []
        total_loss = 0
        
        for class_idx in range(predictions.size(1)):
            if self.ignore_index is not None and class_idx == self.ignore_index:
                continue
                
            pred_class = predictions[:, class_idx]  # [B, D, H, W]
            target_class = targets[:, class_idx]    # [B, D, H, W]
            
            # Compute intersection and union
            intersection = (pred_class * target_class).sum()
            union = pred_class.sum() + target_class.sum()
            
            # Dice coefficient
            dice = (2. * intersection + self.smooth) / (union + self.smooth)
            dice_loss = 1 - dice
            
            # Apply class weights
            if self.weight is not None:
                dice_loss *= self.weight[class_idx]
            
            total_loss += dice_loss
            dice_scores.append(dice.item())
        
        num_classes = len(dice_scores)
        return total_loss / num_classes if num_classes > 0 else total_loss


class CombinedLoss(nn.Module):
    """
    Combine Cross Entropy Loss and Dice Loss
    """
    def __init__(self, ce_weight=0.5, dice_weight=0.5, smooth=1e-6):
        super(CombinedLoss, self).__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss(smooth=smooth)
        
    def forward(self, predictions, targets):
        ce = self.ce_loss(predictions, targets)
        dice = self.dice_loss(predictions, targets)
        return self.ce_weight * ce + self.dice_weight * dice
