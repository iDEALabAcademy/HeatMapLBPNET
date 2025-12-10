"""
Random Projection (RP) Layer Implementation
MAC-free fusion mechanism for LBP bit channels
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


def gate_activate(logit: torch.Tensor, train: bool, use_ste: bool, ste_scale_g: float) -> torch.Tensor:
    """硬前向 + 可选 STE（论文一致）门控激活。

    前向恒为 g_hard = 1_{logit > 0}；若 use_ste=True，则以 sigmoid(ste_scale_g * logit) 仅做梯度替代。
    """
    g_hard = (logit > 0).float()
    if train and use_ste:
        s = max(float(ste_scale_g), 1e-6)
        g_soft = torch.sigmoid(s * logit)
        return g_hard.detach() - g_soft.detach() + g_soft
    return g_hard


class RPLayer(nn.Module):
    """
    Random Projection Layer for MAC-free LBP fusion
    """
    
    def __init__(self, n_bits_per_out: int = 4, n_out_channels: int = 64, seed: int = 42,
                 tau: float = 2.0, learn_tau: bool = False, learnable: bool = True,
                 fusion_type: str = 'rp', bit_select: bool = False,
                 gate_logits_init: float = 0.3, **kwargs):
        super().__init__()
        
        self.n_bits_per_out = n_bits_per_out
        self.n_out_channels = n_out_channels
        self.seed = int(seed)
        self.learnable = bool(learnable)
        self.gate_logits_init = float(gate_logits_init)
        # 当 fusion_type 显式为 'rp' 且 learnable=False 或 bit_select=True 时，走位选择映射
        self.bit_select_mode = bool(bit_select) or (fusion_type == 'rp' and not self.learnable)
        
        # 统一为 tau 接口（兼容旧参数）
        if 'temperature' in kwargs and 'tau' not in kwargs:
            tau = float(kwargs.get('temperature'))
        if 'learn_temperature' in kwargs and 'learn_tau' not in kwargs:
            learn_tau = bool(kwargs.get('learn_temperature'))
        
        # tau 将被弃用，不再影响前向，仅保留兼容接口
        self.learn_tau = bool(learn_tau)
        if self.learn_tau:
            self.tau_param = nn.Parameter(torch.tensor(float(tau)))
        else:
            self.register_buffer('tau_param', torch.tensor(float(tau)))
        # 新增：门控 STE 缩放（仅影响梯度形状）
        ste_scale_g = kwargs.get('ste_scale_g', 6.0)
        self.register_buffer('ste_scale_g', torch.tensor(float(ste_scale_g), dtype=torch.float32))
        
        # 运行时标志
        self._weights_initialized = False
        self.use_ste_gates: bool = True
        
        # 为了 .to(device) 同步，先注册空 buffer 名称
        self.register_buffer('rp_weights', torch.empty(0))
        self.register_buffer('rp_map', torch.empty(0, dtype=torch.long))
        self.gate_logits: Optional[nn.Parameter] = None
        
    @property
    def tau(self) -> float:
        return float(self.tau_param.detach().item())
    
    def set_tau(self, tau: float):
        with torch.no_grad():
            self.tau_param.data.fill_(float(tau))
    
    def set_gate_ste_scale(self, ste_scale_g: float):
        with torch.no_grad():
            self.ste_scale_g.copy_(torch.tensor(float(ste_scale_g), dtype=self.ste_scale_g.dtype, device=self.ste_scale_g.device))
    
    def _init_weights(self, input_bits: int, device: torch.device):
        """在输入相同设备上初始化/重置权重与门控参数，避免设备搬运与全局RNG污染。"""
        if self.bit_select_mode:
            # 仅初始化映射表
            if (self._weights_initialized and self.rp_map.numel() != 0
                and self.rp_map.shape[1] == self.n_bits_per_out and self.rp_map.device == device):
                return
            gen = torch.Generator(device=device); gen.manual_seed(self.seed)
            # 从 total_bits 中选择 n_bits_per_out 的索引（先占位，forward里根据实际 input_bits 重建）
            # 此处仅标记初始化状态，真正尺寸在 forward 校正
            self._weights_initialized = True
            return
        # learnable/dense 投影权重
        if (self._weights_initialized and self.rp_weights.numel() != 0
            and self.rp_weights.shape[1] == input_bits and self.rp_weights.device == device
            and (self.gate_logits is not None) and (self.gate_logits.device == device)):
            return
        
        # 局部随机源（与设备一致），不改全局 RNG
        gen = torch.Generator(device=device)
        gen.manual_seed(self.seed)
        
        weights = torch.randint(
            0, 2, (self.n_out_channels, input_bits), dtype=torch.float32, device=device, generator=gen
        )
        weights = weights * 2 - 1
        weights = weights / np.sqrt(input_bits)
        
        # 重新注册/更新 rp_weights buffer 保持在设备上
        if (self.rp_weights.numel() == 0) or (self.rp_weights.shape != weights.shape) or (self.rp_weights.device != device):
            if hasattr(self, 'rp_weights'):
                delattr(self, 'rp_weights')
            self.register_buffer('rp_weights', weights)
        else:
            with torch.no_grad():
                self.rp_weights.copy_(weights)
        
        # 门控偏置初始化：由 gate_logits_init 控制（默认 +0.3）
        if (self.gate_logits is None) or (self.gate_logits.numel() != input_bits) or (self.gate_logits.device != device):
            init_bias = getattr(self, 'gate_logits_init', 0.3)
            self.gate_logits = nn.Parameter(
                torch.full((input_bits,), float(init_bias), dtype=weights.dtype, device=device),
                requires_grad=True
            )
        
        self._weights_initialized = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, *dims = x.shape
        if len(dims) == 5:  # [B, C, P, N, H, W]
            C, P, N, H, W = dims
            x_reshaped = x.view(B, C * P * N, H, W)
            total_bits = C * P * N
        elif len(dims) == 4:  # [B, P, N, H, W]
            P, N, H, W = dims
            x_reshaped = x.view(B, P * N, H, W)
            total_bits = P * N
        elif len(dims) == 3:  # [B, P*N, H, W]
            total_bits, H, W = dims
            x_reshaped = x
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")
        
        dev = x_reshaped.device
        self._init_weights(total_bits, dev)
        
        # [B, total_bits, H, W] -> [B*H*W, total_bits]
        x_bits = x_reshaped
        x_flat = x_bits.permute(0, 2, 3, 1).contiguous().view(B * H * W, total_bits)

        if self.bit_select_mode:
            # 固定映射：为每个输出通道选择 n_bits_per_out 个输入bit并打包
            if (self.rp_map.numel() == 0) or (self.rp_map.device != dev) or (self.rp_map.shape != (self.n_out_channels, self.n_bits_per_out)):
                gen = torch.Generator(device=dev); gen.manual_seed(self.seed)
                rp_map = torch.randint(0, total_bits, (self.n_out_channels, self.n_bits_per_out), device=dev, dtype=torch.long, generator=gen)
                if hasattr(self, 'rp_map'):
                    delattr(self, 'rp_map')
                self.register_buffer('rp_map', rp_map)
            # gather: [BHW, C_out, n_bits]
            idx = self.rp_map  # [C_out, n_bits]
            x_exp = x_flat.unsqueeze(1).expand(-1, self.n_out_channels, -1)
            gathered = torch.gather(x_exp, 2, idx.unsqueeze(0).expand(x_flat.size(0), -1, -1))
            # bit-pack -> [BHW, C_out]
            weights = (2 ** torch.arange(self.n_bits_per_out, device=dev, dtype=gathered.dtype)).view(1,1,-1)
            packed = (gathered * weights).sum(-1)
            output = packed.view(B, H, W, self.n_out_channels).permute(0, 3, 1, 2).contiguous()
            # 无门控统计时，给出恒定alive统计
            self._last_alive_ratio_soft = 1.0
            self._last_alive_ratio = 1.0
            return output

        # learnable/dense 投影分支（含门控）
        # 先做零中心化（把 [0,1] -> [-1,1]），提升投影稳定性
        x_centered = x_bits * 2.0 - 1.0
        x_flat = x_centered.permute(0, 2, 3, 1).contiguous().view(B * H * W, total_bits)
        # 硬前向 + 可选 STE（不再使用 tau 影响前向）
        gates = gate_activate(self.gate_logits, self.training, self.use_ste_gates, float(self.ste_scale_g.item()))

        gated_x = x_flat * gates.unsqueeze(0)
        output_flat = torch.matmul(gated_x, self.rp_weights.t())
        output = output_flat.view(B, H, W, self.n_out_channels).permute(0, 3, 1, 2)
        self.snapshot_alive()
        return output
    
    def get_gate_values(self) -> Optional[torch.Tensor]:
        if self.gate_logits is None:
            return None
        # 返回软值供正则/监控，但不参与前向决策
        return torch.sigmoid(self.gate_logits)

    def set_gate_requires_grad(self, flag: bool):
        """冻结/解冻门控参数的便捷方法"""
        if hasattr(self, 'gate_logits') and self.gate_logits is not None:
            self.gate_logits.requires_grad_(bool(flag))

    @torch.no_grad()
    def gate_stats(self):
        """返回门控统计信息（软/硬激活、均值/方差）。"""
        if not hasattr(self, 'gate_logits') or self.gate_logits is None:
            return None
        # 使用当前 tau 仅用于形状（不影响前向）
        denom = max(float(self.tau), 1e-6)
        g_soft = torch.sigmoid(self.gate_logits / denom)
        g_hard = (self.gate_logits >= 0).float()
        return {
            'mean_soft': float(g_soft.mean().item()),
            'std_soft': float(g_soft.std(unbiased=False).item()),
            'alive_soft': float(g_soft.mean().item()),
            'alive_hard': float(g_hard.mean().item())
        }
    
    def get_alive_ratio(self) -> float:
        if hasattr(self, '_last_alive_ratio'):
            return self._last_alive_ratio
        gv = self.get_gate_values()
        return 0.0 if gv is None else (gv > 0.5).float().mean().item()
    
    def snapshot_alive(self):
        with torch.no_grad():
            g_soft = torch.sigmoid(self.gate_logits)
            # 仅作调试缓存；报告与决策用硬值
            self._last_alive_ratio_soft = float(g_soft.mean().item())
            self._last_alive_ratio = float(((self.gate_logits >= 0).float().mean()).item())
    
    def extra_repr(self) -> str:
        return (f'n_bits_per_out={self.n_bits_per_out}, '
                f'n_out_channels={self.n_out_channels}, '
                f'seed={self.seed}, '
                f'ste_scale_g={float(self.ste_scale_g.item()):.2f}')
