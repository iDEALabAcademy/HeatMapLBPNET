"""
MAC-free Residual Block Implementation
Combines LBP layer with RP fusion and residual connection
"""

import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any

from ..layers import LBPLayer, RPLayer
from ..layers.rp_paper_layer import RPFusionPaper


class MACFreeBlock(nn.Module):
    """
    MAC-free Residual Block
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        lbp_config (Dict[str, Any]): Configuration for LBP layer
        rp_config (Dict[str, Any]): Configuration for RP layer
        downsample (bool): Whether to downsample spatial dimensions
        use_residual (bool): Whether to use residual connection
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        lbp_config: Dict[str, Any],
        rp_config: Dict[str, Any],
        downsample: bool = False,
        use_residual: bool = True
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.downsample = downsample
        self.use_residual = use_residual
        
        # LBP layer
        self.lbp_layer = LBPLayer(**lbp_config)
        
        # Calculate LBP output channels
        lbp_out_channels = lbp_config['num_patterns'] * lbp_config['num_points']
        
        # Fusion type: 'rp_paper' | 'conv1x1' | 'rp_linear'
        self.fusion_type = rp_config.get('fusion_type', rp_config.get('type', 'rp_paper'))
        lbp_out_channels = lbp_config['num_patterns'] * lbp_config['num_points']
        if self.fusion_type == 'rp_paper':
            self.fuse = RPFusionPaper(
                n_bits_per_out=rp_config.get('n_bits_per_out', 4),
                n_out_channels=out_channels,
                seed=rp_config.get('seed', 42),
                threshold=rp_config.get('threshold', None),
                tau=rp_config.get('tau', 0.5),
                use_ste=rp_config.get('use_ste', True)
            )
        elif self.fusion_type == 'conv1x1':
            self.fuse = nn.Conv2d(lbp_out_channels, out_channels, kernel_size=1, bias=False)
            self.fuse._is_paper_fusion = True
        elif self.fusion_type in ('rp_linear', 'rp'):
            self.fuse = RPLayer(
                n_bits_per_out=rp_config.get('n_bits_per_out', 4),
                n_out_channels=out_channels,
                seed=rp_config.get('seed', 42),
                tau=rp_config.get('tau', rp_config.get('temperature', 2.0)),
                learn_tau=rp_config.get('learn_tau', rp_config.get('learn_temperature', False)),
                learnable=rp_config.get('learnable', True),
                fusion_type='rp',
                bit_select=not rp_config.get('learnable', True)
            )
        else:
            raise ValueError(f"Unknown fusion_type: {self.fusion_type}")

        # Debug bypass: replace RP with 1x1 conv (input channels = P*N); use only when enabled via env var
        self.debug_proj = nn.Conv2d(
            lbp_config['num_patterns'] * lbp_config['num_points'],
            out_channels, kernel_size=1, bias=False
        )
        
        # Batch normalization + activation
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)
    # Residual scaling (improves stability)
        self.res_scale = nn.Parameter(torch.tensor(1.0))
        
        # Downsample layer if needed
        if downsample:
            self.downsample_layer = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # Channel projection for residual connection
        if in_channels != out_channels:
            self.channel_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.channel_proj = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor [B, C_in, H, W]
        
        Returns:
            Output tensor [B, C_out, H, W]
        """
        identity = x
        
        # LBP feature extraction
        lbp_features = self.lbp_layer(x)  # [B, P, N, H, W]

        # Fusion implementation
        if isinstance(self.fuse, nn.Conv2d):
            B, P, N, H, W = lbp_features.shape
            y = lbp_features.reshape(B, P * N, H, W)
            fused_features = self.fuse(y)
        else:
            fused_features = self.fuse(lbp_features)  # [B, C_out, H, W]
        
        # Batch normalization
        fused_features = self.bn(fused_features)
        
        # Downsample if needed
        if self.downsample:
            fused_features = self.downsample_layer(fused_features)
            if self.use_residual:
                identity = self.downsample_layer(identity)
        
        # Residual connection (with residual scaling)
        if self.use_residual:
            if self.channel_proj is not None:
                identity = self.channel_proj(identity)
            output = fused_features * self.res_scale + identity
        else:
            output = fused_features
        
        # Post-activation
        output = self.act(output)
        return output
    
    def get_offset_penalty(self) -> torch.Tensor:
        """Get offset regularization penalty from LBP layer"""
        return self.lbp_layer.get_offset_penalty()
    
    def get_offsets(self) -> torch.Tensor:
        """Get current offsets from LBP layer"""
        return self.lbp_layer.get_offsets()
    
    def get_gate_values(self) -> torch.Tensor:
        """Get gating values from fusion layer if available"""
        if hasattr(self, 'fuse') and hasattr(self.fuse, 'get_gate_values'):
            return self.fuse.get_gate_values()
        return None
    
    def get_alive_ratio(self) -> float:
        """Get alive ratio from fusion layer if available"""
        if hasattr(self, 'fuse') and hasattr(self.fuse, 'get_alive_ratio'):
            return self.fuse.get_alive_ratio()
        return 0.0
    
    def update_alpha(self, alpha: float):
        """Update alpha in LBP layer"""
        self.lbp_layer.update_alpha(alpha)

    def update_tau(self, tau: float):
        """No-op: forward is always hard per the paper; tau no longer affects forward behavior."""
        return

    def set_ste(self, use_ste_bits: bool, use_ste_gates: bool):
        if hasattr(self.lbp_layer, 'set_use_ste_bits'):
            self.lbp_layer.set_use_ste_bits(use_ste_bits)
        if hasattr(self, 'fuse') and hasattr(self.fuse, 'set_use_ste_gates'):
            self.fuse.set_use_ste_gates(use_ste_gates)
    
    def extra_repr(self) -> str:
        return (f'in_channels={self.in_channels}, '
                f'out_channels={self.out_channels}, '
                f'downsample={self.downsample}, '
                f'use_residual={self.use_residual}')

    # ===== Added: gate freezing proxy and stats =====
    def freeze_gates(self, flag: bool):
        """Freeze/unfreeze gating parameters (True = freeze)."""
        # Support rp_linear fusion (RPLayer)
        if hasattr(self, 'fuse') and hasattr(self.fuse, 'set_gate_requires_grad'):
            self.fuse.set_gate_requires_grad(not (not flag))
        # Backward compatibility: older field rp_layer (if present)
        if hasattr(self, 'rp_layer') and hasattr(self.rp_layer, 'set_gate_requires_grad'):
            self.rp_layer.set_gate_requires_grad(not (not flag))

    def gate_stats(self):
        """Return gate statistics (if supported by the fusion implementation)."""
        if hasattr(self, 'fuse') and hasattr(self.fuse, 'gate_stats'):
            return self.fuse.gate_stats()
        if hasattr(self, 'rp_layer') and hasattr(self.rp_layer, 'gate_stats'):
            return self.rp_layer.gate_stats()
        return None
