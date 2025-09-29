import torch
import torch.nn as nn
import torch.nn.functional as F

# 基本卷積模組：DoubleConv (改用 GroupNorm)
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


# 下採樣模組
class Down(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups=8):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),  # 下採樣 2x
            DoubleConv(in_channels, out_channels, num_groups=8, )
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)

# 上採樣模組（修正版）
class Up(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups=8, bilinear=False):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            # bilinear 模式下，輸入通道數就是 in_channels
            self.conv = DoubleConv(in_channels + out_channels, out_channels, num_groups)
        else:
            # 使用反卷積，將 in_channels 減半
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            # concat 後的通道數 = (in_channels // 2) + out_channels
            self.conv = DoubleConv(in_channels // 2 + out_channels, out_channels, num_groups=8)
    
    def forward(self, x1, x2):
        # x1: 來自下層的特徵圖（需要上採樣）
        # x2: 來自編碼器的跳躍連接
        x1 = self.up(x1)
        
        # 如果尺寸不一致，做 padding
        diffZ = x2.size(2) - x1.size(2)
        diffY = x2.size(3) - x1.size(3)
        diffX = x2.size(4) - x1.size(4)
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffZ // 2, diffZ - diffZ // 2])
        
        # 連接跳躍連接和上採樣的特徵圖
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

# 輸出卷積
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        return self.conv(x)