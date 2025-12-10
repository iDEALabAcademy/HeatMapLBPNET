"""
Local Binary Pattern (LBP) Layer Implementation
Learnable LBP feature extraction layer
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Union

# Try to use your packaged heatmap loader if present
try:
    from data.heatmap import get_heatmap_torch  # when project root has 'data' on sys.path
    _HAS_DATA_HEATMAP = True
except Exception:
    try:
        from lbpnet.data.heatmap import get_heatmap_torch  # package-relative fallback
        _HAS_DATA_HEATMAP = True
    except Exception:
        _HAS_DATA_HEATMAP = False

# NEW: adaptive-P helpers (work with either tools.heatmap or data.heatmap)
try:
    from data.heatmap import window_mean_from_heatmap, adaptive_p_mask_from_mean
    _HAS_ADAPTIVE_HELPERS = True
except Exception:
    try:
        from lbpnet.data.heatmap import window_mean_from_heatmap, adaptive_p_mask_from_mean
        _HAS_ADAPTIVE_HELPERS = True
    except Exception:
        _HAS_ADAPTIVE_HELPERS = False


def lbp_binarize(samples: torch.Tensor, anchor: torch.Tensor,
                 ste_scale: Union[float, torch.Tensor], train: bool, use_ste: bool = True) -> torch.Tensor:
    x = samples - anchor
    b_hard = (x > 0).float()
    if train and use_ste:
        s = (ste_scale.clamp(min=1e-6) if isinstance(ste_scale, torch.Tensor)
             else max(float(ste_scale), 1e-6))
        b_soft = torch.sigmoid(s * x)
        return b_hard.detach() - b_soft.detach() + b_soft
    return b_hard


class LBPLayer(nn.Module):
    """
    Local Binary Pattern Layer with learnable sampling offsets.

    Simplified version: removed heatmap-biased offset initialization. Offsets now
    always use Gaussian initialization (clamped to the window radius). We keep an
    optional heatmap_path only for future adaptive-P masking logic; it no longer
    affects offset sampling.

    Adaptive-P masking (optional): if enabled (typically stage 0 only), we compute a
    per-pixel P(h,w) from the local heatmap window mean using configured bins and P
    values. Lower mean → smaller P per your config. We then activate only the first
    P(h,w) neighbors from a fixed, global per-pattern permutation; the rest are zero.
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
        constraint_weight: float = 0.01,
        # ---- (Retained) heatmap path only for adaptive-P mask helpers ----
        heatmap_path: Optional[str] = None,
        # ---- NEW: adaptive-P controls (safe defaults: OFF) ----
        adaptive_p_enable: bool = False,
        adaptive_bins: Tuple[float, float, float] = (0.2, 0.5, 0.7),
        adaptive_values: Tuple[int, ...] = (2, 4, 6, 8),
        adaptive_perm_seed: int = 42,
        is_stage0: bool = False,
        apply_only_stage0: bool = True,
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
        self.alpha_min = float(alpha_min)
        self.tau_min = float(tau_min)

        # Retained for adaptive-P only (no longer influences offset init)
        self.heatmap_path = heatmap_path or os.environ.get("GLOBAL_HEATMAP_PATH", "")

        # NEW: adaptive-P config
        self.adaptive_p_enable = bool(adaptive_p_enable)
        # bins: ascending thresholds; values: len=bins+1; will be clamped to [0, num_points]
        self.adaptive_bins = tuple(map(float, adaptive_bins))
        self.adaptive_values = tuple(int(v) for v in adaptive_values)
        self.adaptive_perm_seed = int(adaptive_perm_seed)
        self.is_stage0 = bool(is_stage0)
        self.apply_only_stage0 = bool(apply_only_stage0)

        # radius (in pixels)
        self.radius = float((window - 1) / 2)

        # params
        self._init_parameters()

        # STE toggle (enabled by default)
        self.use_ste_bits = True
        self._grid_cache = {}

        # cache for adaptive mask per (H,W,device)
        self._adaptive_cache = {}

    # ---------- public toggles ----------

    # (Removed heatmap-biased offset helper methods.)

    # ---------- parameters & numerics ----------
    def _init_parameters(self):
        """Initialize learnable parameters (Gaussian offsets only)."""
        if self.share_across_channels:
            init = torch.randn(self.num_patterns, self.num_points, 2) * float(self.offset_init_std)
            init = torch.clamp(init, -self.radius, self.radius)
            self.offsets_raw = nn.Parameter(init)
        else:
            raise NotImplementedError("Per-channel offsets not implemented yet")

        # ---- Alpha / tau / ste_scale ----
        if self.learn_alpha:
            self.alpha = nn.Parameter(torch.tensor(self.alpha_init))
        else:
            self.register_buffer('alpha', torch.tensor(self.alpha_init))
        self.register_buffer('tau', torch.tensor(3.0, dtype=torch.float32))
        init_scale = float(1.0 / max(float(self.alpha_init), 1e-6))
        self.register_buffer('ste_scale', torch.tensor(init_scale, dtype=torch.float32))

        # ---- Pattern weights ----
        self.pattern_weights = nn.Parameter(torch.ones(self.num_patterns, self.num_points))

    def _get_offsets(self) -> torch.Tensor:
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

    # ---- NEW: adaptive-P mask (cached) ----
    def _get_adaptive_mask(self, H: int, W: int, device: torch.device) -> Optional[torch.Tensor]:
        """Build/fetch adaptive-P mask using a GLOBAL per-pattern permutation (fixed across pixels).

        Behavior:
            1) Compute local window mean (excluding center) and map to P(h,w) via bins/values.
            2) For each pattern p, sample a single random permutation of neighbor indices [0..N-1].
               Keep that permutation fixed spatially.
            3) For each pixel, activate neighbors whose permutation rank < P(h,w).

        Returns mask of shape [num_patterns, num_points, H, W]; applied per-batch in forward.
        """
        if (not self.adaptive_p_enable) or (self.apply_only_stage0 and (not self.is_stage0)):
            return None
        if not _HAS_ADAPTIVE_HELPERS:
            return None

        key = (H, W, device.index if device.type == 'cuda' else -1, 'perm_global_v1', self.adaptive_perm_seed, self.num_points, tuple(self.adaptive_bins), tuple(self.adaptive_values))
        cached = self._adaptive_cache.get(key, None)
        if cached is not None and cached.device == device:
            return cached

        with torch.no_grad():
            # 1) Local window mean
            win_mean = window_mean_from_heatmap(
                path=self.heatmap_path if (self.heatmap_path and len(self.heatmap_path) > 0) else None,
                device=device,
                target_hw=(H, W),
                kernel_size=self.window,
                exclude_center=True
            )  # [1,1,H,W] in [0,1]
            # 2) Derive per-pixel P(h,w) via bins/values mapping
            # Ensure bins sorted
            bins = torch.tensor(self.adaptive_bins, dtype=win_mean.dtype, device=device)
            bins, _ = torch.sort(bins)
            vals = list(self.adaptive_values)
            # Ensure len(vals) == len(bins)+1; if not, fallback to equal splits
            if len(vals) != (bins.numel() + 1):
                # fallback: monotonically decreasing by 2 starting from num_points
                start = int(self.num_points)
                steps = bins.numel() + 1
                vals = [max(1, start - 2 * i) for i in range(steps)]
            vals = torch.tensor(vals, dtype=torch.int64, device=device)
            # Clamp to [0, num_points]
            vals = torch.clamp(vals, min=0, max=int(self.num_points))
            w = win_mean.squeeze(0).squeeze(0)  # [H,W]
            # Build p map piecewise
            p_hw = torch.empty(H, W, dtype=torch.int64, device=device)
            # segment 0: w < bins[0]
            p_hw[:] = vals[-1]  # temporary assign; will overwrite by conditions below
            if bins.numel() > 0:
                mask0 = (w < bins[0])
                p_hw[mask0] = vals[0]
                for i in range(1, bins.numel()):
                    seg = (w >= bins[i-1]) & (w < bins[i])
                    p_hw[seg] = vals[i]
                # last segment: w >= bins[-1]
                p_hw[w >= bins[-1]] = vals[-1]
            else:
                # No bins provided: single value
                p_hw[:] = vals[0]

            # 3) Build GLOBAL permutation ranks per pattern (fixed across H,W)
            gen = torch.Generator(device=device)
            gen.manual_seed(self.adaptive_perm_seed)
            scores = torch.rand(self.num_patterns, self.num_points, generator=gen, device=device)  # [P,N]
            perm = scores.argsort(dim=1, descending=True)  # [P,N] indices ordered by priority
            # Compute rank per neighbor: rank[p, n] = position of n in perm[p]
            rank = torch.empty_like(perm)
            arange_n = torch.arange(self.num_points, device=device).view(1, -1).expand(self.num_patterns, -1)
            rank.scatter_(1, perm, arange_n)  # [P,N], values in [0..N-1]
            # Cache permutation indices for debug/inspection
            try:
                self._global_perm_idx = perm.detach().clone()
            except Exception:
                pass

            # 4) Construct mask by comparing rank against P(h,w)
            full_mask = (rank.unsqueeze(-1).unsqueeze(-1) < p_hw.unsqueeze(0).unsqueeze(1)).float()
            full_mask = full_mask.contiguous()

            self._adaptive_cache[key] = full_mask
            # Expose P map for external use (unchanged API semantics)
            try:
                self._last_adaptive_p_map = p_hw.unsqueeze(0)  # [1,H,W]
            except Exception:
                pass
            return full_mask

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

        # --- NEW: adaptive-P (only if enabled and conditions satisfied) ---
        mask = self._get_adaptive_mask(H=bits.size(-2), W=bits.size(-1), device=bits.device)
        if mask is not None and bits.size(2) == self.num_points:
            # mask shape: [P, N, H, W] -> expand over batch
            m = mask.unsqueeze(0)  # [1,P,N,H,W]
            bits = bits * m  # apply per-pattern, per-pixel neighbor selection
            # Provide broadcasted P map (convert stored single-batch to B,H,W int64)
            try:
                B = bits.size(0)
                p_hw = self._last_adaptive_p_map.squeeze(0).to(torch.int64)  # [H,W]
                self._last_adaptive_p_map = p_hw.unsqueeze(0).expand(B, -1, -1)
            except Exception:
                pass

        # Apply non-negative pattern weights
        w = F.softplus(self.pattern_weights)
        weighted_bits = bits * w.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

        if self.mode == 'bits':
            return bits
        elif self.mode == 'features':
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

    def get_adaptive_permutation(self) -> Optional[torch.Tensor]:
        """Return global permutation indices per pattern [P, N] if cached."""
        return getattr(self, '_global_perm_idx', None)

    def get_last_p_map(self) -> Optional[torch.Tensor]:
        """
        Return the last computed adaptive P map.

        Shape:
            - [B, H, W] if called after a forward pass (batch-expanded), or
            - [1, H, W] if computed via mask precomputation without a forward.

        Returns None if adaptive-P is disabled or a P map hasn't been computed yet.
        """
        return getattr(self, '_last_adaptive_p_map', None)

    def extra_repr(self) -> str:
        alpha_val = float(self.alpha.detach().item())
        ste_scale_val = float(self.ste_scale.detach().item())
        return (f'num_patterns={self.num_patterns}, '
                f'num_points={self.num_points}, '
                f'window={self.window}, '
                f'share_across_channels={self.share_across_channels}, '
                f'mode={self.mode}, '
                f'alpha(compat)={alpha_val:.4f}, '
                f'ste_scale={ste_scale_val:.4f}, '
                f'adaptive_p_enable={self.adaptive_p_enable}, '
                f'is_stage0={self.is_stage0}, '
                f'apply_only_stage0={self.apply_only_stage0})')
