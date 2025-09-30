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

# Imports related to data augmentation
import random
from scipy.ndimage import rotate, zoom, gaussian_filter, map_coordinates
from scipy.ndimage.interpolation import shift
from typing import Tuple, Optional, List, Union

from .augmentation import *

'''
        'light': {
            'description': 'Light augmentation – suitable for small datasets or quick training',
            'transforms': ['Random flip', 'Small-angle rotation (-10°~10°)', 'Slight brightness adjustment', 'Slight contrast adjustment'],
            'use_case': 'Early experiments, small datasets, rapid prototype development'
        },

        'medium': {
            'description': 'Medium augmentation – balance between performance and diversity',
            'transforms': ['Random flip', 'Medium-angle rotation (-15°~15°)', 'Slight scaling', 'Brightness/contrast adjustment', 
                          'Gamma correction', 'Slight noise', 'Slight blur', 'Bias field simulation'],
            'use_case': 'General training scenarios, medium-sized datasets'
        },

        'heavy': {
            'description': 'Heavy augmentation – suitable for large datasets or high generalization requirements',
            'transforms': ['Strong random flip', 'Large-angle rotation (-20°~20°)', 'Elastic deformation', 'Strong brightness/contrast adjustment',
                          'Strong Gamma correction', 'Moderate noise', 'Blur', 'Bias field', 'Ghosting artifact', 'Motion artifact'],
            'use_case': 'Large datasets, models requiring strong generalization ability'
        },

        'medical': {
            'description': 'Medical-specific – preserves anatomical structure',
            'transforms': ['Left-right flip', 'Small-angle rotation (-10°~10°)', 'Slight brightness adjustment', 'Slight contrast adjustment',
                          'Gamma correction', 'Slight noise', 'Bias field', 'Mild ghosting'],
            'use_case': 'Medical image segmentation, tasks requiring anatomical integrity'
        },

        'medical_heavy': {
            'description': 'Heavy medical artifact augmentation – includes various medical imaging artifacts',
            'transforms': ['Left-right flip', 'Medium-angle rotation', 'Brightness/contrast adjustment', 'Gamma correction', 'Moderate noise',
                          'Strong bias field', 'Ghosting artifact', 'Motion artifact', 'Ringing artifact', 'Blur'],
            'use_case': 'Medical image AI models robust to various imaging artifacts'
        }
'''


# ==================== Predefined Data Augmentation Presets ====================
def get_light_augmentation() -> Compose3D:
    """Light augmentation (suitable for small datasets or quick training)"""
    return Compose3D([
        RandomFlip3D(axes=[1, 2], prob=0.5),
        RandomRotation3D(degrees=(-10, 10), prob=0.3),
        RandomBrightness(brightness_range=(-0.05, 0.05), prob=0.3),
        RandomContrast(contrast_range=(0.9, 1.1), prob=0.3),
    ])


def get_medium_augmentation() -> Compose3D:
    """Medium augmentation (balance between performance and diversity)"""
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
    """Heavy augmentation (suitable for large datasets or models requiring strong generalization)"""
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
    """Medical image-specific augmentation (preserves anatomical structure)"""
    return Compose3D([
        RandomFlip3D(axes=[2], prob=0.5),  # only flip left-right
        RandomRotation3D(degrees=(-10, 10), prob=0.4),  # small rotation
        RandomBrightness(brightness_range=(-0.08, 0.08), prob=0.4),
        RandomContrast(contrast_range=(0.85, 1.15), prob=0.4),
        RandomGamma(gamma_range=(0.85, 1.15), prob=0.3),
        RandomNoise(noise_std=0.03, prob=0.3),  # mild noise
        MedicalBiasField(field_strength=0.15, prob=0.2),
        MedicalGhostingArtifact(intensity=0.2, prob=0.1),
    ])


def get_medical_artifact_heavy() -> Compose3D:
    """Heavy medical artifact augmentation (includes various medical imaging artifacts)"""
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
    """Custom augmentation (choose your own transformations)"""
    return Compose3D([
        RandomFlip3D(axes=[1, 2], prob=0.5),
        RandomRotation3D(degrees=(-10, 10), prob=0.3)
    ])

