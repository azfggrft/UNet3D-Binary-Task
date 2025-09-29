
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

from .augmentation import *

'''
        'light': {
            'description': '輕度增強 - 適合小數據集或快速訓練',
            'transforms': ['隨機翻轉', '小角度旋轉(-10°~10°)', '輕微亮度調整', '輕微對比度調整'],
            'use_case': '初期實驗、小型數據集、快速原型開發'
        },

        'medium': {
            'description': '中度增強 - 平衡性能和多樣性',
            'transforms': ['隨機翻轉', '中等角度旋轉(-15°~15°)', '輕微縮放', '亮度/對比度調整', 
                          'Gamma校正', '輕微噪聲', '輕微模糊', '偏置場模擬'],
            'use_case': '一般訓練場景、中等大小數據集'
        },

        'heavy': {
            'description': '重度增強 - 適合大數據集或需要高度泛化',
            'transforms': ['強隨機翻轉', '大角度旋轉(-20°~20°)', '彈性變形', '強亮度/對比度調整',
                          '強Gamma校正', '中等噪聲', '模糊處理', '偏置場', '鬼影偽影', '運動偽影'],
            'use_case': '大型數據集、需要強泛化能力的模型'
        },

        'medical': {
            'description': '醫學專用 - 保持解剖結構完整性',
            'transforms': ['左右翻轉', '小角度旋轉(-10°~10°)', '輕微亮度調整', '輕微對比度調整',
                          'Gamma校正', '輕微噪聲', '偏置場', '輕微鬼影'],
            'use_case': '醫學影像分割、需要保持解剖結構的任務'
        },

        'medical_heavy': {
            'description': '醫學重度偽影 - 包含各種醫學成像偽影',
            'transforms': ['左右翻轉', '中等角度旋轉', '亮度/對比度調整', 'Gamma校正', '中等噪聲',
                          '強偏置場', '鬼影偽影', '運動偽影', '振鈴偽影', '模糊處理'],
            'use_case': '需要對各種成像偽影魯棒的醫學影像AI模型'
        }
'''

# ==================== 預定義的數據增強組合 ====================
def get_light_augmentation() -> Compose3D:
    """輕度數據增強（適合小數據集或快速訓練）"""
    return Compose3D([
        RandomFlip3D(axes=[1, 2], prob=0.5),
        RandomRotation3D(degrees=(-10, 10), prob=0.3),
        RandomBrightness(brightness_range=(-0.05, 0.05), prob=0.3),
        RandomContrast(contrast_range=(0.9, 1.1), prob=0.3),
    ])


def get_medium_augmentation() -> Compose3D:
    """中等數據增強（平衡性能和多樣性）"""
    return Compose3D([
        RandomFlip3D(axes=[1, 2], prob=0.5),
        RandomRotation3D(degrees=(-15, 15), prob=0.5),
        RandomBrightness(brightness_range=(-0.1, 0.1), prob=0.4),
        RandomContrast(contrast_range=(0.8, 1.2), prob=0.4),
        RandomGamma(gamma_range=(0.8, 1.2), prob=0.3),
        RandomNoise(noise_std=0.05, prob=0.3),
        RandomBlur(sigma_range=(0.5, 1.0), prob=0.2),
        MedicalBiasField(field_strength=0.2, prob=0.15),
    ])


def get_heavy_augmentation() -> Compose3D:
    """強度數據增強（適合大數據集或需要高度泛化）"""
    return Compose3D([
        RandomFlip3D(axes=[1, 2], prob=0.6),
        RandomRotation3D(degrees=(-20, 20), prob=0.6),
        RandomBrightness(brightness_range=(-0.15, 0.15), prob=0.5),
        RandomContrast(contrast_range=(0.7, 1.3), prob=0.5),
        RandomGamma(gamma_range=(0.7, 1.3), prob=0.4),
        RandomNoise(noise_std=0.1, prob=0.4),
        RandomBlur(sigma_range=(0.5, 1.5), prob=0.3),
        MedicalBiasField(field_strength=0.3, prob=0.25),
        MedicalGhostingArtifact(intensity=0.3, prob=0.2),
        MedicalMotionArtifact(prob=0.15),
    ])


def get_medical_augmentation() -> Compose3D:
    """醫學影像專用數據增強（保持解剖結構完整性）"""
    return Compose3D([
        RandomFlip3D(axes=[2], prob=0.5),  # 只在左右方向翻轉
        RandomRotation3D(degrees=(-10, 10), prob=0.4),  # 小角度旋轉
        RandomBrightness(brightness_range=(-0.08, 0.08), prob=0.4),
        RandomContrast(contrast_range=(0.85, 1.15), prob=0.4),
        RandomGamma(gamma_range=(0.85, 1.15), prob=0.3),
        RandomNoise(noise_std=0.03, prob=0.3),  # 輕微噪聲
        MedicalBiasField(field_strength=0.15, prob=0.2),
        MedicalGhostingArtifact(intensity=0.2, prob=0.1),
    ])


def get_medical_artifact_heavy() -> Compose3D:
    """醫學影像重度偽影增強（包含各種醫學成像偽影）"""
    return Compose3D([
        RandomFlip3D(axes=[2], prob=0.5),
        RandomRotation3D(degrees=(-12, 12), prob=0.4),
        RandomBrightness(brightness_range=(-0.1, 0.1), prob=0.4),
        RandomContrast(contrast_range=(0.8, 1.2), prob=0.4),
        RandomGamma(gamma_range=(0.8, 1.2), prob=0.3),
        RandomNoise(noise_std=0.05, prob=0.4),
        MedicalBiasField(field_strength=0.4, prob=0.3),
        MedicalGhostingArtifact(intensity=0.4, prob=0.25),
        MedicalMotionArtifact(prob=0.2),
        MedicalRingingArtifact(amplitude=0.2, prob=0.15),
        RandomBlur(sigma_range=(0.5, 1.2), prob=0.2),
    ])

def get_custom_augmentation() -> Compose3D:
    """客制數據加強（可以自行選擇想加入的數據加強）"""
    return Compose3D([
        RandomFlip3D(axes=[1, 2], prob=0.5),
        RandomRotation3D(degrees=(-10, 10), prob=0.3)
    ])
