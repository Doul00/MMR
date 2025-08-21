from torch import nn
from typing import List, Dict
import torchvision.transforms as tvf


def get_transforms(transforms_config: List[Dict]) -> List[nn.Module]:
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
    return tvf.Compose(transforms)


def get_transform(name: str, **kwargs) -> nn.Module:
    """
    Returns a torchvision transform.

    Args:
        name (str): Transform name.

    Returns:
        nn.Module: The transform instance
    """
    if name == 'resize':
        return tvf.Resize((kwargs.get('height', 224), kwargs.get('width', 224)))
    elif name == 'normalize':
        return tvf.Normalize(mean=kwargs.get('mean', [0.485, 0.456, 0.406]),
                             std=kwargs.get('std', [0.229, 0.224, 0.225]))
    elif name == 'to_tensor':
        return tvf.ToTensor()
    else:
        raise ValueError(f"Unknown transform: {name}")