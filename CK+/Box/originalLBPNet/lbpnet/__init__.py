"""
LBPNet: Local Binary Pattern Neural Network
A MAC-free neural network architecture using learnable LBP features
"""

__version__ = "1.0.0"

# Import main components
from .layers import LBPLayer, RPLayer
from .blocks import MACFreeBlock
from .models import LBPNetBase, LBPNetRP, LBPNetConv1x1, build_model
from .data import (
    get_amnist_datasets, 
    get_amnist_dataloaders,
    get_ckplus_datasets,
    get_ckplus_dataloaders,
    get_svhn_datasets,
    get_svhn_dataloaders
)

__all__ = [
    'LBPLayer',
    'RPLayer', 
    'MACFreeBlock',
    'LBPNetBase',
    'LBPNetRP',
    'LBPNetConv1x1',
    'build_model',
    'get_amnist_datasets',
    'get_amnist_dataloaders',
    'get_ckplus_datasets',
    'get_ckplus_dataloaders',
    'get_svhn_datasets',
    'get_svhn_dataloaders'
]
