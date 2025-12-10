"""
LBPNet Conv1x1 Model Implementation
LBPNet with 1x1 convolution fusion (for comparison)
"""

import torch
import torch.nn as nn
from typing import Dict, Any

from .lbpnet_base import LBPNetBase


class LBPNetConv1x1(LBPNetBase):
    """
    LBPNet with 1x1 convolution fusion
    
    This variant uses 1x1 convolutions instead of RP layers
    for comparison purposes.
    """
    
    def __init__(self, config: Dict[str, Any]):
        # Override RP config to use conv1x1
        config = config.copy()
        if 'blocks' in config:
            config['blocks']['fusion_type'] = 'conv1x1'
        
        super().__init__(config)
    
    def forward(self, x):
        """Forward pass - same as base class"""
        return super().forward(x)
