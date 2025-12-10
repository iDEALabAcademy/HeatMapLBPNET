"""
LBPNet RP Model Implementation
LBPNet with Random Projection fusion
"""

from .lbpnet_base import LBPNetBase


class LBPNetRP(LBPNetBase):
    """
    LBPNet with Random Projection fusion
    
    This is the main LBPNet architecture that uses RP layers
    for MAC-free fusion of LBP features.
    """
    
    def __init__(self, config):
        super().__init__(config)
    
    def forward(self, x):
        """Forward pass - same as base class"""
        return super().forward(x)
