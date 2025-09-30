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

# ==================== Data Augmentation Class Definitions ====================
class Transform3D:
    """Base class for 3D medical image data augmentation"""
    
    def __call__(self, image: torch.Tensor, label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            image: image in (C, D, H, W) format
            label: label in (D, H, W) format
        Returns:
            Augmented image and label
        """
        raise NotImplementedError


class RandomRotation3D(Transform3D):
    """3D Random Rotation"""
    
    def __init__(self, degrees: Union[float, Tuple[float, float]] = (-15, 15), 
                 axes: Tuple[int, int] = (1, 2), prob: float = 0.5):
        """
        Args:
            degrees: rotation angle range, either a single value or a (min, max) tuple
            axes: rotation plane, (1, 2) means rotating in the H-W plane
            prob: probability of applying the rotation
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
            
            # Convert to numpy
            image_np = image.numpy()
            label_np = label.numpy()
            
            # Rotate image (each channel separately)
            rotated_image = np.zeros_like(image_np)
            for c in range(image_np.shape[0]):
                rotated_image[c] = rotate(image_np[c], angle, axes=self.axes, 
                                          reshape=False, order=1, mode='constant', cval=0)
            
            # Rotate label (using nearest neighbor interpolation to preserve integers)
            rotated_label = rotate(label_np, angle, axes=self.axes, 
                                   reshape=False, order=0, mode='constant', cval=0)
            
            return torch.from_numpy(rotated_image), torch.from_numpy(rotated_label.astype(np.int64))
        
        return image, label


class RandomFlip3D(Transform3D):
    """3D Random Flip"""
    
    def __init__(self, axes: List[int] = [1, 2], prob: float = 0.5):
        """
        Args:
            axes: axes along which flipping can occur, [1, 2] means H and W axes
            prob: probability of flipping along each axis
        """
        self.axes = axes
        self.prob = prob
    
    def __call__(self, image: torch.Tensor, label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        for axis in self.axes:
            if random.random() < self.prob:
                image = torch.flip(image, dims=[axis])
                label = torch.flip(label, dims=[axis - 1])  # label has one less dimension
        
        return image, label


class RandomNoise(Transform3D):
    """Add random noise"""
    
    def __init__(self, noise_std: float = 0.1, prob: float = 0.5):
        """
        Args:
            noise_std: standard deviation of noise
            prob: probability of applying noise
        """
        self.noise_std = noise_std
        self.prob = prob
    
    def __call__(self, image: torch.Tensor, label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if random.random() < self.prob:
            noise = torch.randn_like(image) * self.noise_std
            image = image + noise
        
        return image, label


class RandomGamma(Transform3D):
    """Random Gamma Correction"""
    
    def __init__(self, gamma_range: Tuple[float, float] = (0.7, 1.3), prob: float = 0.5):
        """
        Args:
            gamma_range: range of gamma values
            prob: probability of applying gamma correction
        """
        self.gamma_range = gamma_range
        self.prob = prob
    
    def __call__(self, image: torch.Tensor, label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if random.random() < self.prob:
            gamma = random.uniform(self.gamma_range[0], self.gamma_range[1])
            # Handle negative values: normalize first
            min_val = image.min()
            if min_val < 0:
                normalized_image = image - min_val
                result = torch.pow(normalized_image, gamma) + min_val
            else:
                result = torch.pow(image, gamma)
            image = result
        
        return image, label


class RandomContrast(Transform3D):
    """Random contrast adjustment"""
    
    def __init__(self, contrast_range: Tuple[float, float] = (0.8, 1.2), prob: float = 0.5):
        """
        Args:
            contrast_range: range for contrast adjustment
            prob: probability of applying contrast adjustment
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
    """Random brightness adjustment"""
    
    def __init__(self, brightness_range: Tuple[float, float] = (-0.1, 0.1), prob: float = 0.5):
        """
        Args:
            brightness_range: range for brightness adjustment
            prob: probability of applying brightness adjustment
        """
        self.brightness_range = brightness_range
        self.prob = prob
    
    def __call__(self, image: torch.Tensor, label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if random.random() < self.prob:
            brightness = random.uniform(self.brightness_range[0], self.brightness_range[1])
            image = image + brightness
        
        return image, label



class RandomBlur(Transform3D):
    """Random Blur"""
    
    def __init__(self, sigma_range: Tuple[float, float] = (0.5, 1.5), prob: float = 0.3):
        """
        Args:
            sigma_range: range of blurring
            prob: probability of applying blur
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
    """Simulate ghosting artifact in medical images"""
    
    def __init__(self, intensity: float = 0.3, shift_range: Tuple[int, int] = (5, 15), prob: float = 0.2):
        """
        Args:
            intensity: ghosting intensity
            shift_range: shift range in pixels
            prob: probability of applying artifact
        """
        self.intensity = intensity
        self.shift_range = shift_range
        self.prob = prob
    
    def __call__(self, image: torch.Tensor, label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if random.random() < self.prob:
            image_np = image.numpy()
            
            shift_pixels = random.randint(self.shift_range[0], self.shift_range[1])
            axis = random.choice([1, 2, 3])  # choose axis to shift
            
            ghost_image = np.zeros_like(image_np)
            for c in range(image_np.shape[0]):
                if axis == 1:  # shift along D-axis
                    ghost_image[c, shift_pixels:, :, :] = image_np[c, :-shift_pixels, :, :]
                elif axis == 2:  # shift along H-axis
                    ghost_image[c, :, shift_pixels:, :] = image_np[c, :, :-shift_pixels, :]
                else:  # shift along W-axis
                    ghost_image[c, :, :, shift_pixels:] = image_np[c, :, :, :-shift_pixels]
            
            result_image = image_np + ghost_image * self.intensity
            return torch.from_numpy(result_image), label
        
        return image, label


class MedicalMotionArtifact(Transform3D):
    """Simulate motion artifact in medical images"""
    
    def __init__(self, blur_kernel_size: int = 5, motion_angle: float = 45, prob: float = 0.15):
        """
        Args:
            blur_kernel_size: size of motion blur kernel
            motion_angle: angle of motion direction
            prob: probability of applying artifact
        """
        self.blur_kernel_size = blur_kernel_size
        self.motion_angle = motion_angle
        self.prob = prob
    
    def __call__(self, image: torch.Tensor, label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if random.random() < self.prob:
            image_np = image.numpy()
            
            kernel = self._create_motion_blur_kernel(self.blur_kernel_size, self.motion_angle)
            
            from scipy.ndimage import convolve
            blurred_image = np.zeros_like(image_np)
            for c in range(image_np.shape[0]):
                for d in range(image_np.shape[1]):
                    blurred_image[c, d] = convolve(image_np[c, d], kernel, mode='constant')
            
            return torch.from_numpy(blurred_image), label
        
        return image, label
    
    def _create_motion_blur_kernel(self, size, angle):
        """Create motion blur kernel"""
        kernel = np.zeros((size, size))
        angle_rad = np.radians(angle)
        
        center = size // 2
        for i in range(-center, center + 1):
            x = center + int(i * np.cos(angle_rad))
            y = center + int(i * np.sin(angle_rad))
            if 0 <= x < size and 0 <= y < size:
                kernel[x, y] = 1
        
        kernel = kernel / np.sum(kernel) if np.sum(kernel) > 0 else kernel
        return kernel


class MedicalRingingArtifact(Transform3D):
    """Simulate ringing (Gibbs) artifact in medical images"""
    
    def __init__(self, frequency: float = 0.1, amplitude: float = 0.2, prob: float = 0.1):
        """
        Args:
            frequency: ringing frequency
            amplitude: ringing amplitude
            prob: probability of applying artifact
        """
        self.frequency = frequency
        self.amplitude = amplitude
        self.prob = prob
    
    def __call__(self, image: torch.Tensor, label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if random.random() < self.prob:
            image_np = image.numpy()
            shape = image_np.shape
            
            slice_idx = random.randint(0, shape[1] - 1)
            h, w = shape[2], shape[3]
            x, y = np.meshgrid(np.arange(w), np.arange(h))
            
            ringing_pattern = self.amplitude * np.sin(2 * np.pi * self.frequency * x) * \
                              np.sin(2 * np.pi * self.frequency * y)
            
            for c in range(shape[0]):
                image_np[c, slice_idx] += ringing_pattern
            
            return torch.from_numpy(image_np), label
        
        return image, label


class MedicalBiasField(Transform3D):
    """Simulate bias field artifact (MRI intensity inhomogeneity)"""
    
    def __init__(self, field_strength: float = 0.3, smoothness: float = 30, prob: float = 0.25):
        """
        Args:
            field_strength: strength of the bias field
            smoothness: smoothness of the field
            prob: probability of applying artifact
        """
        self.field_strength = field_strength
        self.smoothness = smoothness
        self.prob = prob
    
    def __call__(self, image: torch.Tensor, label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if random.random() < self.prob:
            image_np = image.numpy()
            shape = image_np.shape[1:]  # (D, H, W)
            
            bias_field = np.random.randn(*shape)
            bias_field = gaussian_filter(bias_field, sigma=self.smoothness, mode='constant')
            
            bias_field = bias_field / np.std(bias_field) * self.field_strength
            bias_field = np.exp(bias_field)
            
            for c in range(image_np.shape[0]):
                image_np[c] *= bias_field
            
            return torch.from_numpy(image_np), label
        
        return image, label


class Compose3D:
    """Compose multiple 3D transforms"""
    
    def __init__(self, transforms: List[Transform3D]):
        self.transforms = transforms
    
    def __call__(self, image: torch.Tensor, label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        for transform in self.transforms:
            image, label = transform(image, label)
        return image, label




