import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from .net_module import *


# 完整殘差 3D UNet
class UNet3D(nn.Module):
    def __init__(self, n_channels, n_classes, base_channels=64, num_groups=8, bilinear=False):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        # 編碼器部分
        self.inc = ResidualDoubleConv(n_channels, base_channels, num_groups)
        self.down1 = ResDown(base_channels, base_channels * 2, num_groups)
        self.down2 = ResDown(base_channels * 2, base_channels * 4, num_groups)
        self.down3 = ResDown(base_channels * 4, base_channels * 8, num_groups)
        
        # 瓶頸部分
        factor = 2 if bilinear else 1
        self.down4 = ResDown(base_channels * 8, base_channels * 16 // factor, num_groups)
        
        # 解碼器部分
        self.up1 = ResUp(base_channels * 16, base_channels * 8, num_groups, bilinear)
        self.up2 = ResUp(base_channels * 8, base_channels * 4, num_groups, bilinear)
        self.up3 = ResUp(base_channels * 4, base_channels * 2, num_groups, bilinear)
        self.up4 = ResUp(base_channels * 2, base_channels, num_groups, bilinear)
        
        # 輸出層
        self.outc = OutConv(base_channels, n_classes)
    
    def forward(self, x):
        # 編碼器路徑
        x1 = self.inc(x)          # [B, base_channels, D, H, W]
        x2 = self.down1(x1)       # [B, base_channels*2, D/2, H/2, W/2]
        x3 = self.down2(x2)       # [B, base_channels*4, D/4, H/4, W/4]
        x4 = self.down3(x3)       # [B, base_channels*8, D/8, H/8, W/8]
        x5 = self.down4(x4)       # [B, base_channels*16 or 8, D/16, H/16, W/16]
        
        # 解碼器路徑（帶跳躍連接）
        x = self.up1(x5, x4)      # [B, base_channels*8, D/8, H/8, W/8]
        x = self.up2(x, x3)       # [B, base_channels*4, D/4, H/4, W/4]
        x = self.up3(x, x2)       # [B, base_channels*2, D/2, H/2, W/2]
        x = self.up4(x, x1)       # [B, base_channels, D, H, W]
        
        # 最終輸出
        logits = self.outc(x)     # [B, n_classes, D, H, W]
        return logits
    
    def get_model_size(self):
        """計算模型參數量"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params