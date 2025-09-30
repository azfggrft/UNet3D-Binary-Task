import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from .net_module import *


# Complete 3D UNet
class UNet3D(nn.Module):
    def __init__(self, n_channels, n_classes, base_channels=64, num_groups=8, bilinear=False):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Encoder path
        self.inc = DoubleConv(n_channels, base_channels, num_groups)
        self.down1 = Down(base_channels, base_channels * 2, num_groups)
        self.down2 = Down(base_channels * 2, base_channels * 4, num_groups)
        self.down3 = Down(base_channels * 4, base_channels * 8, num_groups)
        
        # Bottleneck
        factor = 2 if bilinear else 1
        self.down4 = Down(base_channels * 8, base_channels * 16 // factor, num_groups)
        
        # Decoder path
        self.up1 = Up(base_channels * 16, base_channels * 8, num_groups, bilinear)
        self.up2 = Up(base_channels * 8, base_channels * 4, num_groups, bilinear)
        self.up3 = Up(base_channels * 4, base_channels * 2, num_groups, bilinear)
        self.up4 = Up(base_channels * 2, base_channels, num_groups, bilinear)
        
        # Output layer
        self.outc = OutConv(base_channels, n_classes)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)          # [B, 32, D, H, W]
        x2 = self.down1(x1)       # [B, 64, D/2, H/2, W/2]
        x3 = self.down2(x2)       # [B, 128, D/4, H/4, W/4]
        x4 = self.down3(x3)       # [B, 256, D/8, H/8, W/8]
        x5 = self.down4(x4)       # [B, 512 or 256, D/16, H/16, W/16]
        
        # Decoder with skip connections
        x = self.up1(x5, x4)      # [B, 256, D/8, H/8, W/8]
        x = self.up2(x, x3)       # [B, 128, D/4, H/4, W/4]
        x = self.up3(x, x2)       # [B, 64, D/2, H/2, W/2]
        x = self.up4(x, x1)       # [B, 32, D, H, W]
        
        # Final output
        logits = self.outc(x)     # [B, n_classes, D, H, W]
        return logits

    def get_model_size(self):
        """Calculate model parameters"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params
