
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

# 數據增強相關導入
import random
from scipy.ndimage import rotate, zoom, gaussian_filter, map_coordinates
from scipy.ndimage.interpolation import shift
from typing import Tuple, Optional, List, Union

#i love you~~~~~~~~~~~~~~

# ==================== 數據增強類別定義 ====================
class Transform3D:
    """3D 醫學影像數據增強基類"""
    
    def __call__(self, image: torch.Tensor, label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            image: (C, D, H, W) 格式的圖像
            label: (D, H, W) 格式的標籤
        Returns:
            增強後的圖像和標籤
        """
        raise NotImplementedError


class RandomRotation3D(Transform3D):
    """3D 隨機旋轉"""
    
    def __init__(self, degrees: Union[float, Tuple[float, float]] = (-15, 15), 
                 axes: Tuple[int, int] = (1, 2), prob: float = 0.5):
        """
        Args:
            degrees: 旋轉角度範圍，可以是單個值或 (min, max) 元組
            axes: 旋轉平面，(1, 2) 表示在 H-W 平面旋轉
            prob: 執行概率
        """
        if isinstance(degrees, (int, float)):
            self.degrees = (-degrees, degrees)
        else:
            self.degrees = degrees
        self.axes = axes
        self.prob = prob
    
    def __call__(self, image: torch.Tensor, label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if random.random() < self.prob:
            angle = random.uniform(self.degrees[0], self.degrees[1])
            
            # 轉換為 numpy
            image_np = image.numpy()
            label_np = label.numpy()
            
            # 旋轉圖像（每個通道分別旋轉）
            rotated_image = np.zeros_like(image_np)
            for c in range(image_np.shape[0]):
                rotated_image[c] = rotate(image_np[c], angle, axes=self.axes, 
                                        reshape=False, order=1, mode='constant', cval=0)
            
            # 旋轉標籤（使用最近鄰插值保持整數值）
            rotated_label = rotate(label_np, angle, axes=self.axes, 
                                 reshape=False, order=0, mode='constant', cval=0)
            
            return torch.from_numpy(rotated_image), torch.from_numpy(rotated_label.astype(np.int64))
        
        return image, label


class RandomFlip3D(Transform3D):
    """3D 隨機翻轉"""
    
    def __init__(self, axes: List[int] = [1, 2], prob: float = 0.5):
        """
        Args:
            axes: 可以翻轉的軸，[1, 2] 表示可以在 H 軸和 W 軸翻轉
            prob: 每個軸的翻轉概率
        """
        self.axes = axes
        self.prob = prob
    
    def __call__(self, image: torch.Tensor, label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        for axis in self.axes:
            if random.random() < self.prob:
                image = torch.flip(image, dims=[axis])
                label = torch.flip(label, dims=[axis - 1])  # label 少一個維度
        
        return image, label


class RandomNoise(Transform3D):
    """添加隨機噪聲"""
    
    def __init__(self, noise_std: float = 0.1, prob: float = 0.5):
        """
        Args:
            noise_std: 噪聲標準差
            prob: 執行概率
        """
        self.noise_std = noise_std
        self.prob = prob
    
    def __call__(self, image: torch.Tensor, label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if random.random() < self.prob:
            noise = torch.randn_like(image) * self.noise_std
            image = image + noise
        
        return image, label


class RandomGamma(Transform3D):
    """隨機 Gamma 校正"""
    
    def __init__(self, gamma_range: Tuple[float, float] = (0.7, 1.3), prob: float = 0.5):
        """
        Args:
            gamma_range: Gamma 值範圍
            prob: 執行概率
        """
        self.gamma_range = gamma_range
        self.prob = prob
    
    def __call__(self, image: torch.Tensor, label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if random.random() < self.prob:
            gamma = random.uniform(self.gamma_range[0], self.gamma_range[1])
            # 處理負值：先正規化到正值範圍
            min_val = image.min()
            if min_val < 0:
                normalized_image = image - min_val
                result = torch.pow(normalized_image, gamma) + min_val
            else:
                result = torch.pow(image, gamma)
            image = result
        
        return image, label


class RandomContrast(Transform3D):
    """隨機對比度調整"""
    
    def __init__(self, contrast_range: Tuple[float, float] = (0.8, 1.2), prob: float = 0.5):
        """
        Args:
            contrast_range: 對比度調整範圍
            prob: 執行概率
        """
        self.contrast_range = contrast_range
        self.prob = prob
    
    def __call__(self, image: torch.Tensor, label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if random.random() < self.prob:
            contrast = random.uniform(self.contrast_range[0], self.contrast_range[1])
            mean = torch.mean(image)
            image = (image - mean) * contrast + mean
        
        return image, label


class RandomBrightness(Transform3D):
    """隨機亮度調整"""
    
    def __init__(self, brightness_range: Tuple[float, float] = (-0.1, 0.1), prob: float = 0.5):
        """
        Args:
            brightness_range: 亮度調整範圍
            prob: 執行概率
        """
        self.brightness_range = brightness_range
        self.prob = prob
    
    def __call__(self, image: torch.Tensor, label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if random.random() < self.prob:
            brightness = random.uniform(self.brightness_range[0], self.brightness_range[1])
            image = image + brightness
        
        return image, label


class RandomBlur(Transform3D):
    """隨機模糊"""
    
    def __init__(self, sigma_range: Tuple[float, float] = (0.5, 1.5), prob: float = 0.3):
        """
        Args:
            sigma_range: 模糊程度範圍
            prob: 執行概率
        """
        self.sigma_range = sigma_range
        self.prob = prob
    
    def __call__(self, image: torch.Tensor, label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if random.random() < self.prob:
            sigma = random.uniform(self.sigma_range[0], self.sigma_range[1])
            
            image_np = image.numpy()
            blurred_image = np.zeros_like(image_np)
            
            for c in range(image_np.shape[0]):
                blurred_image[c] = gaussian_filter(image_np[c], sigma=sigma, mode='constant', cval=0)
            
            return torch.from_numpy(blurred_image), label
        
        return image, label





class MedicalGhostingArtifact(Transform3D):
    """醫學影像鬼影偽影模擬"""
    
    def __init__(self, intensity: float = 0.3, shift_range: Tuple[int, int] = (5, 15), prob: float = 0.2):
        """
        Args:
            intensity: 鬼影強度
            shift_range: 位移範圍（像素）
            prob: 執行概率
        """
        self.intensity = intensity
        self.shift_range = shift_range
        self.prob = prob
    
    def __call__(self, image: torch.Tensor, label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if random.random() < self.prob:
            image_np = image.numpy()
            
            # 隨機選擇位移方向和距離
            shift_pixels = random.randint(self.shift_range[0], self.shift_range[1])
            axis = random.choice([1, 2, 3])  # 選擇位移軸
            
            # 創建鬼影
            ghost_image = np.zeros_like(image_np)
            for c in range(image_np.shape[0]):
                if axis == 1:  # D 軸位移
                    ghost_image[c, shift_pixels:, :, :] = image_np[c, :-shift_pixels, :, :]
                elif axis == 2:  # H 軸位移
                    ghost_image[c, :, shift_pixels:, :] = image_np[c, :, :-shift_pixels, :]
                else:  # W 軸位移
                    ghost_image[c, :, :, shift_pixels:] = image_np[c, :, :, :-shift_pixels]
            
            # 添加鬼影到原始影像
            result_image = image_np + ghost_image * self.intensity
            
            return torch.from_numpy(result_image), label
        
        return image, label


class MedicalMotionArtifact(Transform3D):
    """醫學影像運動偽影模擬"""
    
    def __init__(self, blur_kernel_size: int = 5, motion_angle: float = 45, prob: float = 0.15):
        """
        Args:
            blur_kernel_size: 運動模糊核大小
            motion_angle: 運動方向角度
            prob: 執行概率
        """
        self.blur_kernel_size = blur_kernel_size
        self.motion_angle = motion_angle
        self.prob = prob
    
    def __call__(self, image: torch.Tensor, label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if random.random() < self.prob:
            image_np = image.numpy()
            
            # 創建運動模糊核
            kernel = self._create_motion_blur_kernel(self.blur_kernel_size, self.motion_angle)
            
            # 應用運動模糊
            from scipy.ndimage import convolve
            blurred_image = np.zeros_like(image_np)
            for c in range(image_np.shape[0]):
                for d in range(image_np.shape[1]):
                    blurred_image[c, d] = convolve(image_np[c, d], kernel, mode='constant')
            
            return torch.from_numpy(blurred_image), label
        
        return image, label
    
    def _create_motion_blur_kernel(self, size, angle):
        """創建運動模糊核"""
        kernel = np.zeros((size, size))
        angle_rad = np.radians(angle)
        
        # 計算運動方向
        dx = int(np.cos(angle_rad) * size // 2)
        dy = int(np.sin(angle_rad) * size // 2)
        
        # 創建線性核
        center = size // 2
        for i in range(-center, center + 1):
            x = center + int(i * np.cos(angle_rad))
            y = center + int(i * np.sin(angle_rad))
            if 0 <= x < size and 0 <= y < size:
                kernel[x, y] = 1
        
        # 正規化
        kernel = kernel / np.sum(kernel) if np.sum(kernel) > 0 else kernel
        return kernel


class MedicalRingingArtifact(Transform3D):
    """醫學影像振鈴偽影模擬（Gibbs 振鈴）"""
    
    def __init__(self, frequency: float = 0.1, amplitude: float = 0.2, prob: float = 0.1):
        """
        Args:
            frequency: 振鈴頻率
            amplitude: 振鈴振幅
            prob: 執行概率
        """
        self.frequency = frequency
        self.amplitude = amplitude
        self.prob = prob
    
    def __call__(self, image: torch.Tensor, label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if random.random() < self.prob:
            image_np = image.numpy()
            shape = image_np.shape
            
            # 在隨機切片上添加振鈴偽影
            slice_idx = random.randint(0, shape[1] - 1)
            
            # 創建振鈴模式
            h, w = shape[2], shape[3]
            x, y = np.meshgrid(np.arange(w), np.arange(h))
            
            # 添加正弦波振鈴
            ringing_pattern = self.amplitude * np.sin(2 * np.pi * self.frequency * x) * \
                             np.sin(2 * np.pi * self.frequency * y)
            
            # 應用到影像
            for c in range(shape[0]):
                image_np[c, slice_idx] += ringing_pattern
            
            return torch.from_numpy(image_np), label
        
        return image, label


class MedicalBiasField(Transform3D):
    """醫學影像偏置場偽影模擬（MRI 強度不均勻性）"""
    
    def __init__(self, field_strength: float = 0.3, smoothness: float = 30, prob: float = 0.25):
        """
        Args:
            field_strength: 偏置場強度
            smoothness: 場的平滑度
            prob: 執行概率
        """
        self.field_strength = field_strength
        self.smoothness = smoothness
        self.prob = prob
    
    def __call__(self, image: torch.Tensor, label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if random.random() < self.prob:
            image_np = image.numpy()
            shape = image_np.shape[1:]  # (D, H, W)
            
            # 生成平滑的偏置場
            bias_field = np.random.randn(*shape)
            bias_field = gaussian_filter(bias_field, sigma=self.smoothness, mode='constant')
            
            # 正規化到指定強度
            bias_field = bias_field / np.std(bias_field) * self.field_strength
            bias_field = np.exp(bias_field)  # 轉換為乘性偏置
            
            # 應用偏置場
            for c in range(image_np.shape[0]):
                image_np[c] *= bias_field
            
            return torch.from_numpy(image_np), label
        
        return image, label


class Compose3D:
    """組合多個 3D 變換"""
    
    def __init__(self, transforms: List[Transform3D]):
        self.transforms = transforms
    
    def __call__(self, image: torch.Tensor, label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        for transform in self.transforms:
            image, label = transform(image, label)
        return image, label



