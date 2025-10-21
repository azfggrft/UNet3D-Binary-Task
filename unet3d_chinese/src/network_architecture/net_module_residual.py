import torch
import torch.nn as nn
import torch.nn.functional as F

# 殘差卷積模組
class ResidualDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups=8):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups, out_channels)
        )
        
        # 如果輸入輸出通道數不同，使用 1x1 卷積調整
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1),
                nn.GroupNorm(num_groups, out_channels)
            )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = self.shortcut(x)
        out = self.double_conv(x)
        out += residual  # 殘差連接
        out = self.relu(out)
        return out


# 殘差下採樣模組
class ResDown(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups=8):
        super().__init__()
        self.maxpool = nn.MaxPool3d(2)
        self.conv = ResidualDoubleConv(in_channels, out_channels, num_groups)
    
    def forward(self, x):
        x = self.maxpool(x)
        return self.conv(x)


# 殘差上採樣模組
class ResUp(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups=8, bilinear=False):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = ResidualDoubleConv(in_channels + out_channels, out_channels, num_groups)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = ResidualDoubleConv(in_channels // 2 + out_channels, out_channels, num_groups)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # 處理尺寸不一致
        diffZ = x2.size(2) - x1.size(2)
        diffY = x2.size(3) - x1.size(3)
        diffX = x2.size(4) - x1.size(4)
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffZ // 2, diffZ - diffZ // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


# 輸出卷積
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        return self.conv(x)