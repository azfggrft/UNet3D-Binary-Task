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
    計算每個前景類別的 Dice 分數，回傳 list。
    class 0 視為背景，跳過。

    Returns:
        list of float, 長度 = num_classes - 1
        e.g. num_classes=4 → [dice_cls1, dice_cls2, dice_cls3]
    """
    if predictions.dim() == 5:
        predictions = torch.argmax(predictions, dim=1)  # [B, D, H, W]

    dice_scores = []
    for cls in range(1, num_classes):          # 跳過背景 class 0
        pred_mask   = (predictions == cls).float()
        target_mask = (targets     == cls).float()

        intersection = (pred_mask * target_mask).sum()
        union        = pred_mask.sum() + target_mask.sum()

        dice = (2. * intersection + smooth) / (union + smooth)
        dice_scores.append(dice.item())

    return dice_scores                         # [dice_cls1, dice_cls2, ...]


def calculate_metrics(predictions, targets, num_classes):
    """
    回傳 accuracy、每個前景類別 Dice、以及所有前景的平均 Dice。
    """
    if predictions.dim() == 5:
        predictions = torch.argmax(predictions, dim=1)

    accuracy   = (predictions == targets).float().mean().item()
    dice_list  = calculate_dice_score(predictions, targets, num_classes)
    mean_dice  = float(np.mean(dice_list)) if dice_list else 0.0

    return {
        'accuracy':        accuracy,
        'mean_dice':       mean_dice,          # 所有前景的平均
        'dice_per_class':  dice_list,          # [cls1, cls2, cls3, ...]
        'foreground_dice': mean_dice,          # 向後相容
    }
