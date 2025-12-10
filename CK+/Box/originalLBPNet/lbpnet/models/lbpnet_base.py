"""
LBPNet Base Model Implementation
Base class for all LBPNet architectures
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional

from ..blocks import MACFreeBlock


class LBPNetBase(nn.Module):
    """
    Base LBPNet model
    
    Args:
        config (Dict[str, Any]): Model configuration
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        self.config = config
        self.num_classes = config['head']['num_classes']
        
        # Extract configurations
        lbp_config = config['lbp_layer']
        block_config = config['blocks']
        head_config = config['head']
        
        # Build model components
        self._build_stem(lbp_config)
        self._build_stages(block_config, lbp_config)
        self._build_head(head_config)
        
        # Initialize weights
        self._init_weights()
    
    def _build_stem(self, lbp_config: Dict[str, Any]):
        """Build input stem: 让 LBP 直接看到原图，避免前置空间混合"""
        self.stem = nn.Identity()
        self._stem_out_ch = 1  # MNIST灰度
    
    def _build_stages(self, block_config: Dict[str, Any], lbp_config: Dict[str, Any]):
        """Build network stages"""
        stages = []
        # 从 stem 输出通道开始
        in_channels = getattr(self, '_stem_out_ch', 1)
        
        for i, out_channels in enumerate(block_config['channels_per_stage']):
            # Check if downsample at this stage
            downsample = i in block_config.get('downsample_at', [])
            
            # Create MAC-free block
            # 合并全局 rp_layer 配置（如 gate_logits_init 等）
            rp_cfg = dict(block_config.get('rp_config', {}))
            rp_cfg.update(self.config.get('rp_layer', {}))

            # 为 LBP 层合并硬化下限（alpha_min/tau_min）
            lbp_cfg = dict(lbp_config)
            hardening = self.config.get('hardening', {})
            if 'alpha_min' in hardening:
                lbp_cfg['alpha_min'] = hardening.get('alpha_min', 0.12)
            if 'tau_min' in hardening:
                lbp_cfg['tau_min'] = hardening.get('tau_min', 1.2)

            block = MACFreeBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                lbp_config=lbp_cfg,
                rp_config=rp_cfg,
                downsample=downsample,
                use_residual=True
            )
            
            stages.append(block)
            in_channels = out_channels
        
        self.stages = nn.ModuleList(stages)
    
    def _build_head(self, head_config: Dict[str, Any]):
        """Build classification head"""
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Flatten
        self.flatten = nn.Flatten()
        
        # Fully connected layers
        use_bn = head_config.get('use_bn', False)
        layers = []
        layers.append(nn.Linear(self.stages[-1].out_channels, head_config['hidden']))
        if use_bn:
            layers.append(nn.BatchNorm1d(head_config['hidden']))
        layers.append(nn.ReLU(inplace=True))
        if head_config.get('dropout_rate', 0.0) and head_config['dropout_rate'] > 0:
            layers.append(nn.Dropout(head_config['dropout_rate']))
        layers.append(nn.Linear(head_config['hidden'], self.num_classes))
        self.fc_layers = nn.Sequential(*layers)
    
    def _init_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor [B, 1, H, W]
        
        Returns:
            Logits [B, num_classes]
        """
        # Stem
        x = self.stem(x)
        
        # Stages
        for stage in self.stages:
            x = stage(x)
        
        # Head
        x = self.global_pool(x)
        x = self.flatten(x)
        x = self.fc_layers(x)
        
        return x
    
    def get_offset_penalty(self) -> torch.Tensor:
        """Get total offset regularization penalty"""
        total_penalty = torch.tensor(0.0, device=next(self.parameters()).device)
        
        for stage in self.stages:
            total_penalty = total_penalty + stage.get_offset_penalty()
        
        return total_penalty
    
    def collect_offsets_from_model(self, device: torch.device) -> Dict[str, torch.Tensor]:
        """Collect offsets from all LBP layers for visualization"""
        offsets_dict = {}
        
        for i, stage in enumerate(self.stages):
            offsets = stage.get_offsets()
            if offsets is not None:
                offsets_dict[f'stage_{i}'] = offsets.to(device)
        
        return offsets_dict
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'lbp_config': self.config['lbp_layer'],
            'block_config': self.config['blocks'],
            'head_config': self.config['head'],
            'config_source': self.config.get('_source', 'unknown')
        }
    
    def update_alpha(self, alpha: float):
        """Update alpha in all LBP layers"""
        for stage in self.stages:
            stage.update_alpha(alpha)

    def update_tau(self, tau: float):
        """No-op（论文一致硬前向设计，不再使用 tau 影响前向）。"""
        return

    def set_ste(self, use_ste_bits: bool, use_ste_gates: bool):
        for stage in self.stages:
            if hasattr(stage, 'set_ste'):
                stage.set_ste(use_ste_bits, use_ste_gates)

    # ===== 新增：批量门控冻结/参数收集/统计 =====
    def freeze_gates(self, flag: bool):
        for stage in self.stages:
            if hasattr(stage, 'freeze_gates'):
                stage.freeze_gates(flag)

    def collect_gate_params(self):
        params = []
        for stage in self.stages:
            # 兼容 fuse 或 rp_layer
            if hasattr(stage, 'fuse') and hasattr(stage.fuse, 'gate_logits') and stage.fuse.gate_logits is not None:
                params.append(stage.fuse.gate_logits)
            if hasattr(stage, 'rp_layer') and hasattr(stage.rp_layer, 'gate_logits') and stage.rp_layer.gate_logits is not None:
                params.append(stage.rp_layer.gate_logits)
        return params

    def collect_offset_params(self):
        params = []
        for stage in self.stages:
            if hasattr(stage, 'lbp_layer'):
                for name, p in stage.lbp_layer.named_parameters(recurse=True):
                    if ('offset' in name) and p.requires_grad:
                        params.append(p)
        return params

    @torch.no_grad()
    def gate_stats(self):
        out = []
        for i, stage in enumerate(self.stages):
            if hasattr(stage, 'gate_stats'):
                s = stage.gate_stats()
                if s is not None:
                    s['stage'] = i
                    out.append(s)
        return out
