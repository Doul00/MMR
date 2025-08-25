"""
@brief: Image transform utils. Reimplements some transforms to handle joint masks and images augmentations.
"""

import torch
import random
from torch import nn
from typing import List, Dict
import torchvision.transforms as tvf


class RandomGaussianBlur(nn.Module):
    def __init__(self, p: float = 0.5, kernel_sizes: List[int] = [3]):
        super().__init__()
        self.p = p
        self.kernel_sizes = kernel_sizes
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        if random.random() < self.p:
            kernel_size = random.choice(self.kernel_sizes)
            x = tvf.GaussianBlur(kernel_size)(x)
        if mask is not None:
            return x, mask
        else:
            return x


class RandomColorJitter(nn.Module):
    def __init__(self, p: float = 0.5, brightness: float = 0.1, contrast: float = 0.1, saturation: float = 0.1, hue: float = 0.1):
        super().__init__()
        self.p = p
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        if random.random() < self.p:
            x = tvf.ColorJitter(self.brightness, self.contrast, self.saturation, self.hue)(x)

        if mask is not None:
            return x, mask
        else:
            return x


class RandomHorizontalFlip(nn.Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        if random.random() < self.p:
            x = tvf.RandomHorizontalFlip(p=1.0)(x)
            if mask is not None:
                mask = tvf.RandomHorizontalFlip(p=1.0)(mask)

        if mask is not None:
            return x, mask
        else:
            return x

class RandomVerticalFlip(nn.Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        if random.random() < self.p:
            x = tvf.RandomVerticalFlip(p=1.0)(x)
            if mask is not None:
                mask = tvf.RandomVerticalFlip(p=1.0)(mask)

        if mask is not None:
            return x, mask
        else:
            return x
        
class Normalize(nn.Module):
    def __init__(self, mean: float = 0.5, std: float = 0.5):
        super().__init__()
        self.mean = mean
        self.std = std
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        x = tvf.Normalize(self.mean, self.std)(x)
        if mask is not None:
            return x, mask
        else:
            return x
        
class Resize(nn.Module):
    def __init__(self, size: int, interpolation: str = 'bilinear'):
        super().__init__()
        self.size = size

        if interpolation == 'nearest':
            interpolation = tvf.InterpolationMode.NEAREST
        elif interpolation == 'bilinear':
            interpolation = tvf.InterpolationMode.BILINEAR
        elif interpolation == 'bicubic':
            interpolation = tvf.InterpolationMode.BICUBIC
        self.interpolation = interpolation
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        x = tvf.Resize(self.size, self.interpolation)(x)
        if mask is not None:
            mask = tvf.Resize(self.size, tvf.InterpolationMode.NEAREST)(mask)
            return x, mask
        else:
            return x

class ToTensor(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        x = tvf.ToTensor()(x)
        if mask is not None:
            # Do not rescale mask values
            mask = tvf.PILToTensor()(mask)
            return x, mask
        else:
            return x


class Compose(nn.Module):
    def __init__(self, transforms: List[nn.Module]):
        super().__init__()
        self.transforms = transforms
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        for transform in self.transforms:
            x, mask = transform(x, mask)
        return x, mask


def get_transforms(transforms_config: List[Dict]) -> nn.Module:
    """
    Get a list of torchvision transforms.

    Args:
        transforms_config(List[Dict]): List of dictionaries with the parameters for each transform.

    Returns:
        list: List of torchvision transforms.
    """
    transforms = []
    for cfg in transforms_config:
        transforms.append(get_transform(**cfg))
    return Compose(transforms)
        

def get_transform(name: str, **kwargs) -> nn.Module:
    """
    Returns a torchvision transform.

    Args:
        name (str): Transform name.

    Returns:
        nn.Module: The transform instance
    """
    if name == 'resize':
        return Resize(**kwargs)
    elif name == 'normalize':
        return Normalize(**kwargs)
    elif name == 'to_tensor':
        return ToTensor()
    # Augmentations
    elif name == 'random_horizontal_flip':
        return RandomHorizontalFlip(**kwargs)
    elif name == 'random_vertical_flip':
        return RandomVerticalFlip(**kwargs)
    elif name == 'random_gaussian_blur':
        return RandomGaussianBlur(**kwargs)
    elif name == 'random_color_jitter':
        return RandomColorJitter(**kwargs)
    else:
        raise ValueError(f"Unknown transform: {name}")