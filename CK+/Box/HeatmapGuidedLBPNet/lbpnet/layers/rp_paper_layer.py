"""
Paper-style Random Projection Fusion: bit routing + popcount + threshold (no MAC)
"""

import math
import numpy as np
import torch
import torch.nn as nn
from typing import Optional


def ste_binarize(x: torch.Tensor, threshold: float, ste_scale_g: float, train: bool, use_ste: bool) -> torch.Tensor:
    """硬前向 + 可选 STE 梯度替代：前向恒以 (x>=threshold) 的硬比较输出。

    若 use_ste=True，使用 sigmoid(ste_scale_g * (x - threshold)) 仅提供梯度形状。
    """
    y_hard = (x >= threshold).float()
    if train and use_ste:
        s = max(float(ste_scale_g), 1e-6)
        y_soft = torch.sigmoid(s * (x - threshold))
        return y_hard.detach() - y_soft.detach() + y_soft
    return y_hard


class RPFusionPaper(nn.Module):
    """
    论文口径的随机投影融合：位路由 + popcount + 阈值（无乘法）
    输入: [B, P, N, H, W] 或 [B, P*N, H, W]
    输出: [B, C_out, H, W]
    """

    def __init__(
        self,
        n_bits_per_out: int = 4,
        n_out_channels: int = 64,
        seed: int = 42,
        threshold: Optional[int] = None,
        tau: float = 0.5,
        use_ste: bool = True,
    ) -> None:
        super().__init__()
        self.k = int(n_bits_per_out)
        self.C_out = int(n_out_channels)
        self.seed = int(seed)
        # tau 仅保留兼容；前向硬，不再影响数值
        self.tau = float(tau)
        self.use_ste = bool(use_ste)
        self._initialized: bool = False
        self._last_alive_ratio: float = 0.0

        # 运行时根据输入bit维度初始化
        self.register_buffer('rp_map_idx', torch.empty(0, dtype=torch.long))
        self.threshold = int(threshold) if (threshold is not None) else int(math.ceil(self.k / 2))
        # 门控 STE 缩放（仅影响梯度形状）
        self.register_buffer('ste_scale_g', torch.tensor(6.0, dtype=torch.float32))

        # 供“论文口径”统计识别
        self._is_paper_fusion = True

    def _init_map(self, total_bits: int, device: torch.device) -> None:
        if self._initialized and self.rp_map_idx.numel() == self.C_out * self.k and getattr(self, '_total_bits', None) == total_bits:
            return
        rng = np.random.RandomState(self.seed)
        idx = []
        for _ in range(self.C_out):
            idx.append(rng.choice(total_bits, size=self.k, replace=False))
        idx = np.stack(idx, axis=0)  # [C_out, k]
        idx_t = torch.from_numpy(idx).long().to(device)
        if self.rp_map_idx.numel() == 0:
            self.register_buffer('rp_map_idx', idx_t)
        else:
            self.rp_map_idx = idx_t
        self._total_bits = int(total_bits)
        self._initialized = True

    @torch.no_grad()
    def _snapshot_alive(self, s: torch.Tensor) -> None:
        # Add debug checks
        if torch.isnan(s).any():
            print(f"WARNING: NaN values detected in input tensor")
            self._last_alive_ratio = 0.0
            self._last_alive_ratio_soft = 0.0
            return
        if not torch.isfinite(s).all():
            print(f"WARNING: Inf values detected in input tensor")
            self._last_alive_ratio = 0.0
            self._last_alive_ratio_soft = 0.0
            return
            
        try:
            # 以硬阈值统计 alive ratio；可选缓存软值做调试（不打印）
            threshold_mask = (s >= self.threshold).float()
            self._last_alive_ratio = float(threshold_mask.mean().item())
            
            scale = float(self.ste_scale_g.item())
            scaled_diff = scale * (s - self.threshold)
            # Clamp values to avoid numerical instability
            scaled_diff = torch.clamp(scaled_diff, min=-20.0, max=20.0)
            y_soft = torch.sigmoid(scaled_diff)
            self._last_alive_ratio_soft = float(y_soft.mean().item())
        except Exception as e:
            print(f"WARNING: Error in _snapshot_alive: {str(e)}")
            self._last_alive_ratio = 0.0
            self._last_alive_ratio_soft = 0.0

    def set_tau(self, tau: float) -> None:
        # 兼容接口：不再影响前向，仅保留
        self.tau = float(tau)

    def set_gate_ste_scale(self, ste_scale_g: float) -> None:
        with torch.no_grad():
            self.ste_scale_g.copy_(torch.tensor(float(ste_scale_g), dtype=self.ste_scale_g.dtype, device=self.ste_scale_g.device))

    def set_use_ste_gates(self, flag: bool) -> None:
        self.use_ste = bool(flag)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, *dims = x.shape
        if len(dims) == 4:  # [B, P, N, H, W]
            P, N, H, W = dims
            bits = x.view(B, P * N, H, W)
        elif len(dims) == 3:  # [B, P*N, H, W]
            bits = x
            _, H, W = dims
        else:
            raise ValueError(f'unexpected input shape {x.shape}')

        total_bits = bits.shape[1]
        self._init_map(total_bits, bits.device)

        # [B, H, W, total_bits]
        t = bits.permute(0, 2, 3, 1).contiguous()
        # 选择位: [B, H, W, C_out, k]
        sel = t[..., self.rp_map_idx]
        # popcount（浮点加法）: [B, H, W, C_out]
        s = sel.sum(dim=-1)

        # 记录活跃度
        with torch.no_grad():
            self._snapshot_alive(s)

        # 硬前向 + 可选 STE 梯度替代（不使用 tau 影响前向）
        y = ste_binarize(s, self.threshold, float(self.ste_scale_g.item()), self.training, self.use_ste)
        y = y.permute(0, 3, 1, 2).contiguous()  # [B, C_out, H, W]
        return y

    def get_gate_values(self):  # 兼容接口
        return None

    def get_alive_ratio(self) -> float:
        return getattr(self, '_last_alive_ratio', 0.0)



