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

# ==================== Dice Loss 實作 ====================
class DiceLoss(nn.Module):
    """
    Dice Loss 實作，適用於多類別分割
    """
    def __init__(self, smooth=1e-6, ignore_index=None, weight=None):
        """
        Args:
            smooth: 平滑項，避免除零
            ignore_index: 忽略的類別索引（如背景）
            weight: 各類別權重
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.weight = weight
        
    def forward(self, predictions, targets):
        """
        Args:
            predictions: [B, C, D, H, W] 模型輸出 logits
            targets: [B, D, H, W] 或 [B, C, D, H, W] 真實標籤
        """
        # 將 logits 轉換為機率分布
        predictions = F.softmax(predictions, dim=1)
        
        # 如果 targets 是 [B, D, H, W] 格式，轉換為 one-hot
        if targets.dim() == 4:
            # 檢查標籤值範圍是否正確
            min_val = targets.min().item()
            max_val = targets.max().item()
            num_classes = predictions.size(1)
            
            if min_val < 0:
                print(f"警告: 發現負數標籤值 {min_val}，將其設為 0")
                targets = torch.clamp(targets, min=0)
            
            if max_val >= num_classes:
                print(f"警告: 標籤值 {max_val} 超出類別範圍 [0, {num_classes-1}]，將其截取")
                targets = torch.clamp(targets, max=num_classes-1)
            
            # 轉換為 one-hot 編碼
            targets = F.one_hot(targets.long(), num_classes=num_classes)
            targets = targets.permute(0, 4, 1, 2, 3).float()  # [B, C, D, H, W]
        
        # 計算每個類別的 Dice 係數
        dice_scores = []
        total_loss = 0
        
        for class_idx in range(predictions.size(1)):
            if self.ignore_index is not None and class_idx == self.ignore_index:
                continue
                
            pred_class = predictions[:, class_idx]  # [B, D, H, W]
            target_class = targets[:, class_idx]    # [B, D, H, W]
            
            # 計算交集和聯集
            intersection = (pred_class * target_class).sum()
            union = pred_class.sum() + target_class.sum()
            
            # Dice 係數
            dice = (2. * intersection + self.smooth) / (union + self.smooth)
            dice_loss = 1 - dice
            
            # 應用權重
            if self.weight is not None:
                dice_loss *= self.weight[class_idx]
            
            total_loss += dice_loss
            dice_scores.append(dice.item())
        
        num_classes = len(dice_scores)
        return total_loss / num_classes if num_classes > 0 else total_loss

class CombinedLoss(nn.Module):
    """
    結合 Cross Entropy 和 Dice Loss
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