import torch
import torch.nn as nn
import torch.nn.functional as F

# Basic Convolution Module: DoubleConv (using GroupNorm)
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups=8):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups, out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


# Downsampling Module
class Down(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups=8):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),  # Downsample by 2x
            DoubleConv(in_channels, out_channels, num_groups=8)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)

# Upsampling Module (Revised)
class Up(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups=8, bilinear=False):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            # In bilinear mode, the input channels remain as in_channels
            self.conv = DoubleConv(in_channels + out_channels, out_channels, num_groups)
        else:
            # Use transposed convolution and halve in_channels
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            # Channels after concatenation = (in_channels // 2) + out_channels
            self.conv = DoubleConv(in_channels // 2 + out_channels, out_channels, num_groups=8)
    
    def forward(self, x1, x2):
        # x1: features from the previous layer (to be upsampled)
        # x2: skip connection from the encoder
        x1 = self.up(x1)
        
        # If sizes do not match, apply padding
        diffZ = x2.size(2) - x1.size(2)
        diffY = x2.size(3) - x1.size(3)
        diffX = x2.size(4) - x1.size(4)
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffZ // 2, diffZ - diffZ // 2])
        
        # Concatenate skip connection and upsampled features
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

# Output Convolution
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        return self.conv(x)
