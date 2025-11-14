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


from .loss import DiceLoss,CombinedLoss

# ==================== Dice 分數計算 ====================
def calculate_dice_score(predictions, targets, num_classes, smooth=1e-6):
    """
    計算 Dice 分數 - 僅計算前景類別（標籤為 1）
    
    Args:
        predictions: [B, C, D, H, W] 或 [B, D, H, W] 預測結果
        targets: [B, D, H, W] 真實標籤
        num_classes: 類別數量
        smooth: 平滑項
        
    Returns:
        dice_score: 前景類別的 Dice 分數 (float)
    """
    # 如果 predictions 有 channel 維度，取 argmax
    if predictions.dim() == 5:
        predictions = torch.argmax(predictions, dim=1)  # [B, D, H, W]
    
    # 只計算前景類別（標籤為 1）
    class_idx = 1
    
    # 創建二進制掩碼 - 只針對前景類別
    pred_mask = (predictions == class_idx).float()
    target_mask = (targets == class_idx).float()
    
    # 計算交集和聯集
    intersection = (pred_mask * target_mask).sum()
    union = pred_mask.sum() + target_mask.sum()
    
    # 計算 Dice 分數
    if union == 0:
        dice = 1.0 if intersection == 0 else 0.0
    else:
        dice = (2. * intersection + smooth) / (union + smooth)
    
    return float(dice)

def calculate_metrics(predictions, targets, num_classes):
    """
    計算多種評估指標 - 針對前景類別優化
    """
    if predictions.dim() == 5:
        predictions = torch.argmax(predictions, dim=1)
    
    # 整體準確率
    accuracy = (predictions == targets).float().mean().item()
    
    # 前景類別 Dice 分數
    dice_score = calculate_dice_score(predictions, targets, num_classes)
    
    return {
        'accuracy': accuracy,
        'mean_dice': dice_score,
        'foreground_dice': dice_score,
        'dice_per_class': [dice_score]  # 保持向後兼容
    }
