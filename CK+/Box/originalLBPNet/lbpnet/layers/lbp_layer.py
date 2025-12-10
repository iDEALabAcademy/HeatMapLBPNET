"""
Local Binary Pattern (LBP) Layer Implementation
Learnable LBP feature extraction layer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Union


def lbp_binarize(samples: torch.Tensor, anchor: torch.Tensor,
                 ste_scale: Union[float, torch.Tensor], train: bool, use_ste: bool = True) -> torch.Tensor:
    """Hard-forward LBP comparison with optional STE gradient surrogate (consistent with the paper).

    Forward is always a hard threshold: b_hard = 1_{(samples - anchor) > 0}.
    If use_ste=True, sigmoid(ste_scale * (samples - anchor)) is used only to shape gradients.
    """
    # Compute difference
    x = samples - anchor
    # Hard forward
    b_hard = (x > 0).float()
    if train and use_ste:
        # Surrogate used only for gradients
        if isinstance(ste_scale, torch.Tensor):
            s = torch.clamp(ste_scale, min=1e-6)
        else:
            s = max(float(ste_scale), 1e-6)
        b_soft = torch.sigmoid(s * x)
        return b_hard.detach() - b_soft.detach() + b_soft
    return b_hard


class LBPLayer(nn.Module):
    """
    Local Binary Pattern Layer with learnable sampling offsets
    
    Args:
        num_patterns (int): Number of LBP patterns
        num_points (int): Number of sampling points per pattern
        window (int): Window size for LBP computation
        share_across_channels (bool): Whether to share offsets across channels
        mode (str): Output mode ('bits', 'features', 'both')
        alpha_init (float): Initial alpha value for soft comparison
        learn_alpha (bool): Whether to learn alpha parameter
        offset_init_std (float): Standard deviation for offset initialization
        use_soft_constraint (bool): Whether to use soft radius constraint
        target_radius (Optional[float]): Target radius for soft constraint
        constraint_weight (float): Weight for radius constraint
    """
    
    def __init__(
        self,
        num_patterns: int = 1,
        num_points: int = 8,
        window: int = 5,
        share_across_channels: bool = True,
        mode: str = 'bits',
        alpha_init: float = 0.2,
        learn_alpha: bool = True,
        offset_init_std: float = 0.3,
        alpha_min: float = 0.12,
        tau_min: float = 1.2,
        use_soft_constraint: bool = False,
        target_radius: Optional[float] = None,
        constraint_weight: float = 0.01
    ):
        super().__init__()
        
        self.num_patterns = num_patterns
        self.num_points = num_points
        self.window = window
        self.share_across_channels = share_across_channels
        self.mode = mode
        self.alpha_init = alpha_init
        self.learn_alpha = learn_alpha
        self.offset_init_std = offset_init_std
        self.use_soft_constraint = use_soft_constraint
        self.target_radius = target_radius
        self.constraint_weight = constraint_weight
    # Lower bounds for hardening
        self.alpha_min = float(alpha_min)
        self.tau_min = float(tau_min)
        
        # Calculate radius
        self.radius = float((window - 1) / 2)
        
    # Initialize learnable parameters
        self._init_parameters()
    # STE toggle (enabled by default)
        self.use_ste_bits: bool = True
        # base grid cache
        self._grid_cache = {}
    
    def _init_parameters(self):
        """Initialize learnable parameters"""
        # Initialize offsets
        if self.share_across_channels:
            # Use offset_init_std to control the initial offset distribution (in pixel units), then clamp within the window radius
            init = torch.randn(self.num_patterns, self.num_points, 2) * float(self.offset_init_std)
            init = torch.clamp(init, -self.radius, self.radius)
            self.offsets_raw = nn.Parameter(init)
        else:
            # Separate offsets for each channel (not implemented in this version)
            raise NotImplementedError("Per-channel offsets not implemented yet")
        
        # Keep alpha parameter/buffer for compatibility (not used in forward);
        # introduce ste_scale (affects gradient shape only). By default, ste_scale = 1/alpha.
        if self.learn_alpha:
            self.alpha = nn.Parameter(torch.tensor(self.alpha_init))
        else:
            self.register_buffer('alpha', torch.tensor(self.alpha_init))
        # Compatibility: add tau parameter for the LBP layer (not used in forward numerics; for unified interface and logging)
        self.register_buffer('tau', torch.tensor(3.0, dtype=torch.float32))
        # ste_scale as a buffer (excluded from weight decay); initialize to 1/max(alpha, eps)
        init_scale = float(1.0 / max(float(self.alpha_init), 1e-6))
        self.register_buffer('ste_scale', torch.tensor(init_scale, dtype=torch.float32))
        
        # Initialize pattern weights
        self.pattern_weights = nn.Parameter(
            torch.ones(self.num_patterns, self.num_points)
        )
    
    def _get_offsets(self) -> torch.Tensor:
        """Continuous mapping of offsets to avoid zero gradients at the boundaries"""
        raw = self.offsets_raw
        offsets = self.radius * torch.tanh(raw / max(self.radius, 1e-6))
        return torch.clamp(offsets, -self.radius, self.radius)

    def _get_base_grid(self, H: int, W: int, device: torch.device):
        key = (H, W, device)
        if key in self._grid_cache:
            return self._grid_cache[key]
        gy, gx = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing='ij'
        )
        self._grid_cache[key] = (gy, gx)
        return gy, gx
    
    def _compute_bits(self, x: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        """
        Compute LBP bits using grid sampling
        
        Args:
            x: Input tensor [B, C, H, W]
            offsets: Sampling offsets [P, N, 2]
        
        Returns:
            LBP bits [B, P, N, H, W]
        """
        B, C, H, W = x.shape
        device = x.device
        
        # Create coordinate grid
        grid_y, grid_x = self._get_base_grid(H, W, device)
        
        # Expand for batch and pattern dimensions
        grid_x = grid_x.unsqueeze(0).unsqueeze(0).expand(B, self.num_patterns, H, W)
        grid_y = grid_y.unsqueeze(0).unsqueeze(0).expand(B, self.num_patterns, H, W)
        
        # Apply offsets to grid coordinates
        offsets_x = offsets[:, :, 0].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # [1, P, N, 1, 1]
        offsets_y = offsets[:, :, 1].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # [1, P, N, 1, 1]
        
        # Convert pixel offsets to normalized coordinates
        offsets_x_norm = offsets_x / (W - 1) * 2
        offsets_y_norm = offsets_y / (H - 1) * 2
        
        # Apply offsets
        sample_x = grid_x.unsqueeze(2) + offsets_x_norm  # [B, P, N, H, W]
        sample_y = grid_y.unsqueeze(2) + offsets_y_norm  # [B, P, N, H, W]
        
        # Stack coordinates for grid_sample
        sample_grid = torch.stack([sample_x, sample_y], dim=-1)  # [B, P, N, H, W, 2]
        
        # Reshape for grid_sample
        sample_grid_flat = sample_grid.view(B * self.num_patterns * self.num_points, H, W, 2)
        x_expanded = x.unsqueeze(1).unsqueeze(1).expand(B, self.num_patterns, self.num_points, C, H, W)
        x_flat = x_expanded.reshape(B * self.num_patterns * self.num_points, C, H, W)
        
        # Sample neighbor values
        sampled_neighbors = F.grid_sample(
            x_flat, sample_grid_flat, 
            mode='bilinear', 
            align_corners=True,
            padding_mode='border'
        )  # [B*P*N, C, H, W]
        
        # Reshape back
        sampled_neighbors = sampled_neighbors.view(B, self.num_patterns, self.num_points, C, H, W)
        
        # Get center pixel values (reference)
        center_values = x.unsqueeze(1).unsqueeze(1).expand(B, self.num_patterns, self.num_points, C, H, W)
        
        # Hard forward + optional STE (gradient surrogate only)
        bits = lbp_binarize(
            samples=sampled_neighbors,
            anchor=center_values,
            ste_scale=self.ste_scale,
            train=self.training,
            use_ste=self.use_ste_bits
        )
        
        return bits
    
    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass
        
        Args:
            x: Input tensor [B, C, H, W]
        
        Returns:
            LBP features or (bits, features) depending on mode
        """
    # Apply lower bounds at runtime as well for numerical stability
        self.apply_hardening_floor_()
        # Get current offsets
        offsets = self._get_offsets()
        
        # Compute LBP bits
        bits = self._compute_bits(x, offsets)
        
        # Reduce channel dimension if present: [B, P, N, C, H, W] -> [B, P, N, H, W]
        if bits.dim() == 6:
            bits = bits.mean(dim=3)
        
        # Apply non-negative pattern weights
        w = F.softplus(self.pattern_weights)
        weighted_bits = bits * w.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        
        if self.mode == 'bits':
            # As per the paper: output raw binary bits to the fusion layer (unweighted)
            return bits
        elif self.mode == 'features':
            # Convert bits to features
            features = weighted_bits.sum(dim=2)  # Sum over sampling points
            return features
        elif self.mode == 'both':
            return bits, weighted_bits.sum(dim=2)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
    
    def get_offset_penalty(self) -> torch.Tensor:
        """Symmetric radius regularization (r - target)^2"""
        if self.target_radius is None:
            return torch.zeros((), device=self.offsets_raw.device)
        offsets = self._get_offsets()
        r = offsets.norm(dim=-1)
        penalty = (r - self.target_radius).pow(2)
        return penalty.mean() * self.constraint_weight
    
    def update_alpha(self, alpha: float):
        """Update alpha and synchronize ste_scale = 1/max(alpha, eps) (affects gradient shape only; hard forward unchanged)."""
        a = torch.as_tensor(alpha, dtype=self.alpha.dtype, device=self.alpha.device)
        with torch.no_grad():
            self.alpha.copy_(a)
            # Synchronize ste_scale
            scale = 1.0 / max(float(a.item()), 1e-6)
            self.ste_scale.copy_(torch.tensor(scale, dtype=self.ste_scale.dtype, device=self.ste_scale.device))
        # Apply lower bounds
        self.apply_hardening_floor_()

    @torch.no_grad()
    def set_tau_(self, tau_value: float):
        if hasattr(self, 'tau') and isinstance(self.tau, torch.Tensor):
            self.tau.data.fill_(float(tau_value))
        self.apply_hardening_floor_()

    @torch.no_grad()
    def set_alpha_(self, alpha_value: float):
        if hasattr(self, 'alpha') and isinstance(self.alpha, torch.Tensor):
            self.alpha.data.fill_(float(alpha_value))
            # Synchronize ste_scale
            scale = 1.0 / max(float(alpha_value), 1e-6)
            self.ste_scale.data.copy_(torch.tensor(scale, dtype=self.ste_scale.dtype, device=self.ste_scale.device))
        self.apply_hardening_floor_()

    @torch.no_grad()
    def apply_hardening_floor_(self):
        if hasattr(self, 'alpha') and isinstance(self.alpha, torch.Tensor):
            self.alpha.data.clamp_(min=float(self.alpha_min))
        if hasattr(self, 'tau') and isinstance(self.tau, torch.Tensor):
            self.tau.data.clamp_(min=float(self.tau_min))

    def set_use_ste_bits(self, flag: bool):
        """Enable/disable STE for LBP bits."""
        self.use_ste_bits = bool(flag)
    
    def get_offsets(self) -> torch.Tensor:
        """Get current offsets for visualization"""
        return self._get_offsets().detach()
    
    def extra_repr(self) -> str:
        alpha_val = float(self.alpha.detach().item())
        ste_scale_val = float(self.ste_scale.detach().item())
        return (f'num_patterns={self.num_patterns}, '
                f'num_points={self.num_points}, '
                f'window={self.window}, '
                f'share_across_channels={self.share_across_channels}, '
                f'mode={self.mode}, '
                f'alpha(compat)={alpha_val:.4f}, '
                f'ste_scale={ste_scale_val:.4f}')
