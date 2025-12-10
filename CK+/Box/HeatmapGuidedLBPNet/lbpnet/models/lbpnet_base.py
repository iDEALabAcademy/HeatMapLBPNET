"""
LBPNet Base Model Implementation
Base class for all LBPNet architectures
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional

from ..blocks import MACFreeBlock


class LBPNetBase(nn.Module):
    """
    Base LBPNet model

    Expects config sections:
      config['lbp_layer'] : dict of defaults for all LBP layers
      config['blocks']    : {
          'channels_per_stage': [int, ...],
          'downsample_at': [int, ...],
          'rp_config': {...}   # optional per-block RP defaults
      }
      config['head']      : {'num_classes': int, 'hidden': int, ...}

        Optional:
            config['hardening'] = {
          'alpha_min': 0.12,
          'tau_min': 1.2
      }

      # Global RP overrides (merged into blocks.rp_config):
      config['rp_layer'] : {...}

      # NEW: Adaptive-P preset (forwarded into LBPLayer kwargs)
      config['adaptive_p'] = {
          'enable': False,
          'apply_only_stage0': True,     # if False, applies to all stages
          'thresholds': [0.2, 0.5, 0.7], # three cut points → four bins
          'values': [2, 4, 6, 8],        # P per bin (len must be 4)
          'kernel': 5,                   # window size used for mean
          # optional per-model override; if omitted, LBPLayer uses its own
          # heatmap_path or the global heatmap.path if set for stage 0
          'heatmap_path': None
      }
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
        """Build input stem: let LBP see the raw image (grayscale)."""
        self.stem = nn.Identity()
        self._stem_out_ch = 1  # grayscale input

    def _mk_lbp_cfg_for_stage(self, base_lbp_cfg: Dict[str, Any], stage_idx: int) -> Dict[str, Any]:
        """
        Clone and specialize LBP layer config per stage.
        - Enforce hardening floors (alpha_min/tau_min) if provided.
    - Thread adaptive-P preset if requested.
        """
        lbp_cfg = copy.deepcopy(base_lbp_cfg)

        # ---- Optional per-stage num_points decay (takes highest precedence) ----
        # If the user supplies a list `num_points_per_stage`, we override num_points for each stage.
        # Stage 0 will therefore automatically get the (possibly larger) first entry; we then
        # drop any `num_points_stage0` hint to avoid a second override later.
        try:
            per_stage_list = base_lbp_cfg.get('num_points_per_stage', None)
            if per_stage_list and isinstance(per_stage_list, (list, tuple)):
                if stage_idx < len(per_stage_list):
                    lbp_cfg['num_points'] = int(per_stage_list[stage_idx])
                    if 'num_points_stage0' in lbp_cfg:
                        lbp_cfg.pop('num_points_stage0', None)
        except Exception:
            pass

        # ---- Hardening floors (kept for compatibility with older training scripts)
        hardening = self.config.get('hardening', {})
        if 'alpha_min' in hardening:
            lbp_cfg['alpha_min'] = float(hardening['alpha_min'])
        if 'tau_min' in hardening:
            lbp_cfg['tau_min'] = float(hardening['tau_min'])

        # (Removed heatmap-based offset init logic.)

        # ---- Stage-0 override for num_points (allow Pmax on first layer)
        # (Skip if already provided by per-stage list above)
        if stage_idx == 0 and 'num_points_stage0' in lbp_cfg:
            try:
                if 'num_points_per_stage' not in base_lbp_cfg:
                    lbp_cfg['num_points'] = int(lbp_cfg.pop('num_points_stage0'))
                else:
                    lbp_cfg.pop('num_points_stage0', None)
            except Exception:
                lbp_cfg.pop('num_points_stage0', None)

        # ---- Adaptive-P preset threading (non-invasive; only adds kwargs)
        ap = self.config.get('adaptive_p', None)
        if ap and bool(ap.get('enable', False)):
            apply_only_stage0 = bool(ap.get('apply_only_stage0', True))
            if (stage_idx == 0) or (not apply_only_stage0):
                # Validate minimal fields; let LBPLayer do deeper checks
                lbp_cfg['adaptive_p_enable'] = True
                lbp_cfg['adaptive_p_apply_only_stage0'] = bool(ap.get('apply_only_stage0', True))
                if 'thresholds' in ap:
                    lbp_cfg['adaptive_p_thresholds'] = list(ap['thresholds'])
                if 'values' in ap:
                    lbp_cfg['adaptive_p_values'] = list(ap['values'])
                if 'kernel' in ap:
                    lbp_cfg['adaptive_p_kernel'] = int(ap['kernel'])
                # Optional explicit heatmap path for adaptive-P (window mean computation)
                if ap.get('heatmap_path', None) is not None:
                    lbp_cfg['heatmap_path'] = ap['heatmap_path']

        return lbp_cfg

    def _build_stages(self, block_config: Dict[str, Any], lbp_config: Dict[str, Any]):
        """Build network stages"""
        stages = []
        in_channels = getattr(self, '_stem_out_ch', 1)

        for i, out_channels in enumerate(block_config['channels_per_stage']):
            # Downsample after this stage?
            downsample = i in block_config.get('downsample_at', [])

            # Merge RP configs: block-level defaults overridden by global rp_layer (if any)
            rp_cfg = dict(block_config.get('rp_config', {}))
            rp_cfg.update(self.config.get('rp_layer', {}))

            # Per-stage LBP config (handles layer1-only heatmap + hardening + adaptive-P)
            lbp_cfg = self._mk_lbp_cfg_for_stage(lbp_config, i)

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
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

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
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.global_pool(x)
        x = self.flatten(x)
        x = self.fc_layers(x)
        return x

    def get_offset_penalty(self) -> torch.Tensor:
        """Total offset regularization penalty (sum over stages)."""
        total_penalty = torch.tensor(0.0, device=next(self.parameters()).device)
        for stage in self.stages:
            total_penalty = total_penalty + stage.get_offset_penalty()
        return total_penalty

    def collect_offsets_from_model(self, device: torch.device) -> Dict[str, torch.Tensor]:
        """Collect offsets from all LBP layers for visualization."""
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
        """No-op: forward is always hard per the paper."""
        return

    def set_ste(self, use_ste_bits: bool, use_ste_gates: bool):
        for stage in self.stages:
            if hasattr(stage, 'set_ste'):
                stage.set_ste(use_ste_bits, use_ste_gates)

    # ===== gate freezing / param collectors / stats =====
    def freeze_gates(self, flag: bool):
        for stage in self.stages:
            if hasattr(stage, 'freeze_gates'):
                stage.freeze_gates(flag)

    def collect_gate_params(self):
        params = []
        for stage in self.stages:
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
