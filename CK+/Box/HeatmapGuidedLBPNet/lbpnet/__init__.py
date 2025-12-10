"""
LBPNet: Local Binary Pattern Neural Network
A MAC-free neural network architecture using learnable LBP features
"""

__version__ = "1.0.0"

# Import main components
from .layers import LBPLayer, RPLayer
from .blocks import MACFreeBlock
from .models import (
    LBPNetBase,
    LBPNetRP,
    LBPNetConv1x1,
    build_model,
    get_default_config
)
from .data import get_amnist_datasets, get_amnist_dataloaders

# Backwards-compatible aliases
get_mnist_datasets = get_amnist_datasets
get_mnist_dataloaders = get_amnist_dataloaders
get_svhn_datasets = get_amnist_datasets
get_svhn_dataloaders = get_amnist_dataloaders

__all__ = [
    'LBPLayer',
    'RPLayer',
    'MACFreeBlock',
    'LBPNetBase',
    'LBPNetRP',
    'LBPNetConv1x1',
    'build_model',
    'get_default_config',
    'get_mnist_datasets',
    'get_mnist_dataloaders'
]
