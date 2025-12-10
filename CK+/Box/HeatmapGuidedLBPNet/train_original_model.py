#!/usr/bin/env python3
"""
Train LBPNet with original LBP layer on SVHN
Adapted for SVHN dataset and grayscale conversion
"""

import os
import copy
from collections import deque
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import numpy as np
import matplotlib.pyplot as plt
import time
import yaml
import csv

from lbpnet.models import build_model
from tools.metrics_paper import get_paper_model_size_bytes, estimate_ops_paper
from lbpnet.data import get_svhn_datasets

def anneal(epoch, max_epoch, start, end, mode="exp"):
    if mode == "exp":
        r = (end / max(start, 1e-8)) ** (epoch / max(1, max_epoch))
        return start * r
    t = epoch / max(1, max_epoch)
    return start + (end - start) * t

# General interpolation (more readable)
def interp(v0, v1, t, mode="exp"):
    t = float(max(0.0, min(1.0, t)))
    if mode == "exp":
        v0 = max(1e-8, float(v0)); v1 = max(1e-8, float(v1))
        return v0 * ((v1 / v0) ** t)
    return v0 + (v1 - v0) * t


def get_image_hw(cfg) -> Tuple[int, int]:
    """Return (H, W) from cfg['image_size'] which may be int or (h, w)."""
    sz = cfg.get('image_size', 32)
    if isinstance(sz, (list, tuple)) and len(sz) == 2:
        return int(sz[0]), int(sz[1])
    return int(sz), int(sz)


class TemperatureScheduler:
    """Unified manager for alpha/tau annealing and collapse-guard softening.

    - Supports cosine (default) or exponential.
    - After validation, can temporarily soften gates (increase tau) and freeze alpha for a few steps when collapse is detected.
    - Note: in our implementation tau maps to the STE scale for gates: ste_scale_g = 1 / max(tau, 1e-6).
    """

    def __init__(self, total_epochs: int,
                 alpha_start: float = 1.5, alpha_min: float = 0.06,
                 tau_start: float = 3.0, tau_min: float = 0.75,
                 mode: str = "cosine",
                 guard_freeze_alpha_epochs: int = 0):
        self.total_epochs = int(total_epochs)
        self.alpha_start = float(alpha_start)
        self.alpha_min = float(alpha_min)
        self.tau_start = float(tau_start)
        self.tau_min = float(tau_min)
        self.mode = str(mode)

        self._epoch = 0
        self._guard_events = 0
        self._freeze_alpha_steps = int(guard_freeze_alpha_epochs)

        # Dynamic softening multiplier (>1 temporarily raises tau)
        self._tau_soften_multiplier = 1.0

        # Sliding window over val acc
        self._acc_window = deque(maxlen=5)

    def _cosine(self, start: float, end: float, t: float) -> float:
        import math
        t = max(0.0, min(1.0, t))
        return end + (start - end) * 0.5 * (1 + math.cos(math.pi * t))

    def step(self, epoch: Optional[int] = None) -> Tuple[float, float]:
        if epoch is not None:
            self._epoch = int(epoch)
        t = self._epoch / max(1, self.total_epochs - 1)
        if self.mode == "cosine":
            alpha = self._cosine(self.alpha_start, self.alpha_min, t)
            tau = self._cosine(self.tau_start, self.tau_min, t)
        else:
            alpha = interp(self.alpha_start, self.alpha_min, t, mode="exp")
            tau = interp(self.tau_start, self.tau_min, t, mode="exp")

        # If within freeze window, keep alpha from dropping too low
        if self._freeze_alpha_steps > 0:
            alpha = max(alpha, self.alpha_min * 1.5)  # slightly above min to avoid over-hardening
            self._freeze_alpha_steps -= 1

        # Apply softening multiplier to tau (temporarily raises tau -> softer gate STE)
        tau = min(self.tau_start, tau * self._tau_soften_multiplier)

        return float(alpha), float(tau)

    def update_on_val(self, val_acc: float, alive_hard: float,
                      acc_drop_thr: float = 0.05, alive_upper: float = 0.26) -> None:
        # Record acc
        self._acc_window.append(float(val_acc))
        if len(self._acc_window) < self._acc_window.maxlen:
            return
        avg5 = sum(self._acc_window) / len(self._acc_window)
        # Collapse condition: too-high activity or sharp relative accuracy drop
        if (alive_hard is not None and alive_hard > float(alive_upper)) or (self._acc_window[-1] < (1.0 - float(acc_drop_thr)) * avg5):
            # Trigger guard: temporarily increase tau and freeze alpha annealing for a few steps
            self._tau_soften_multiplier = min(self._tau_soften_multiplier * 1.08, 2.0)
            self._freeze_alpha_steps = max(self._freeze_alpha_steps, 3)
            self._guard_events += 1

    def soften(self, extra_freeze_alpha_steps: int = 3, tau_cap: Optional[float] = None):
        self._tau_soften_multiplier = min(self._tau_soften_multiplier * 1.25, 3.0)
        self._freeze_alpha_steps = max(self._freeze_alpha_steps, int(extra_freeze_alpha_steps))
        if tau_cap is not None:
            # Approximate cap via multiplier limit
            self._tau_soften_multiplier = min(self._tau_soften_multiplier, tau_cap / max(1e-6, self.tau_min))
        self._guard_events += 1

    def state_dict(self) -> dict:
        return {
            'epoch': self._epoch,
            'guard_events': self._guard_events,
            'freeze_alpha_steps': self._freeze_alpha_steps,
            'tau_soften_multiplier': self._tau_soften_multiplier,
            'acc_window': list(self._acc_window),
            'cfg': {
                'total_epochs': self.total_epochs,
                'alpha_start': self.alpha_start, 'alpha_min': self.alpha_min,
                'tau_start': self.tau_start, 'tau_min': self.tau_min,
                'mode': self.mode,
            }
        }

    def load_state_dict(self, state: dict):
        if not state:
            return
        self._epoch = int(state.get('epoch', 0))
        self._guard_events = int(state.get('guard_events', 0))
        self._freeze_alpha_steps = int(state.get('freeze_alpha_steps', 0))
        self._tau_soften_multiplier = float(state.get('tau_soften_multiplier', 1.0))
        self._acc_window.clear()
        for v in state.get('acc_window', [])[-self._acc_window.maxlen:]:
            self._acc_window.append(float(v))

    @property
    def guard_events(self) -> int:
        return int(self._guard_events)


class ModelEMA:
    """Simple EMA: shadow = decay * shadow + (1-decay) * param"""
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().clone()

    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            if name in self.shadow and param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    def apply_to(self, model: nn.Module) -> dict:
        """Load EMA weights into the model and return a backup of original weights."""
        backup = {}
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self.shadow and param.requires_grad:
                    backup[name] = param.detach().clone()
                    param.copy_(self.shadow[name])
        return backup

    @staticmethod
    def restore(model: nn.Module, backup: dict):
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in backup:
                    param.copy_(backup[name])

def set_bn_mode(model, mode: str):
    if mode == "freeze":
        for m in model.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False
    elif mode == "track":
        for m in model.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.train()
    else:
        return

PRESETS = {
    "paper_svhn_rp": {
        # model selection (string -> model builder in lbpnet.models)
        "model": "lbpnet_rp",

        # data-splitting / dataset-level options
        "data": {
            "train_ratio": 0.85,   # fraction of data used for training (rest for val)
            "val_ratio": 0.15,     # fraction of data used for validation
            "seed": 42             # split RNG seed for reproducibility
        },

        # expected input image size (SVHN is 32x32)
        "image_size": 32,

        # LBP layer configuration
        "lbp_layer": {
            "num_patterns": 2,            # how many patterns per LBP unit (paper uses small P)
            "num_points": 8,              # sampling points per pattern (e.g., 8 for 3x3 window)
            "window": 5,                  # kernel window (3 => 3x3 neighborhood)
            "share_across_channels": True,# share sampling offsets across channels to save params
            "mode": "bits",               # output representation: "bits" (binary/comparisons)
            "alpha_init": 0.2,            # initial sharpening / slope for soft comparisons
            "learn_alpha": True,          # whether alpha is a learned parameter
            "offset_init_std": 0.02       # stddev for offset initialization (smaller for SVHN)
        },

        # Block/stage configuration (how many stages, channels, where to downsample)
        "blocks": {
            "stages": 8,                      # number of stages
            "channels_per_stage": [37, 40, 80, 80, 160, 160, 320, 320], # output channels per stage (paper-like)
            "downsample_at": [1,3],           # stage indices after which to downsample (0-based)
            "fusion_type": "rp_paper",        # how LBP outputs are fused: 'rp_paper' uses RP mapping
            # RP-specific mapping config (if fusion_type uses rp)
            "rp_config": {
                "fusion_type": "rp_paper",
                "n_bits_per_out": 8,      # number of bits used per output (RP output width)
                "seed": 42,               # seed used to construct random projection mapping
                "threshold": None,        # per-bit threshold (None => use default/learned behavior)
                "tau": 0.5,               # initial tau for gates / numeric mapping (paper-specific)
                "use_ste": True          # whether to use Straight-Through Estimator for RP mapping
            }
        },

        # Classification head config
        "head": {
            "hidden": 512,        # hidden dimension for classifier head(s)
            "dropout_rate": 0.25,  # dropout probability (0 => disabled)
            "num_classes": 10,    # SVHN has 10 classes (digits 0-9)
            "use_bn": True        # whether to use BatchNorm in classifier head
        },

        # High-level training hyperparameters (these are the paper-like choices)
        "training": {
            "epochs": 220,             # total training epochs
            "batch_size": 128,         # per-device batch size
            "lr": 1e-3,                # base learning rate (used when optim.lr not provided)
            "weight_decay": 0.01,      # base weight decay
            "lr_scheduler": "cosine",  # lr schedule: "cosine" or "step"
            "warmup_epochs": 5,        # warmup duration in epochs (if scheduler supports it)
            "patience": 80,            # early stopping patience (if used)
            "min_delta": 5e-4,         # minimum improvement threshold for early stopping
            "label_smoothing": 0.1     # label smoothing factor for CrossEntropyLoss
        },

        # STE settings (how the Straight-Through Estimator is applied)
        "ste": {
            "use_ste_bits": True,      # apply STE for bit outputs
            "use_ste_gates": True,     # apply STE for gate outputs
            "lbp_ste_scale": 6.0,      # multiplicative scale for bit STE (numerical stability)
            "gate_ste_scale": 6.0      # multiplicative scale for gate STE
        },

        # Optimizer micro-config (different lr/weight-decay multipliers are applied later)
        "optim": {
            "type": "AdamW",         # "AdamW" recommended for stability
            "lr": 1e-3,              # initial lr for optimizer (used to build param groups)
            "weight_decay": 1e-2,    # base weight decay for optimizer
            "lr_mult": {             # multipliers applied to param groups (gates/offsets/base)
                "gates": 2.0,
                "offsets": 2.0,
                "base": 1.0
            }
        },

        # Unified temp (alpha/tau) schedule used by TemperatureScheduler
        "temp_schedule": {
            "alpha": {"start": 1.5, "end": 0.06, "mode": "cosine"},  # alpha anneal (hardness)
            "tau":   {"start": 3.0, "end": 0.75, "mode": "cosine"}   # tau anneal (gate softness)
        },

        # Freeze schedule for offsets/gates during training/hard-finetune
        "freeze_schedule": {
            "freeze_offsets_epochs": 5,    # freeze offset learning for first N epochs
            "freeze_gates_extra_epochs": 2,# extra epochs to keep gates frozen (guard)
            "freeze_gate_epochs": 8        # gate freeze period used during training schedule
        },

        # Stability / collapse-guard monitoring settings
        "stability": {
            "collapse_guard": True,    # enable collapse monitoring (recommended)
            "collapse_window": 5,      # how many epochs to look back for collapse detection
            "acc_drop_thr": 0.05,      # relative accuracy drop threshold to trigger guard
            "alive_upper": 0.26        # upper bound for alive ratio to consider "over-active"
        },

        # EMA: Exponential Moving Average of model params for smoother evaluation
        "ema": {"enable": True, "decay": 0.999},

        # Data augmentation (disabled)
        "augment": {
            "enable": False,
            "crop_pad": 2,
            "rotation": 10
        },

        # Parameter-group learning multipliers used when constructing the optimizer
        "optimizer_groups": {
            "lbp_offsets": {"lr_multiplier": 5.0, "weight_decay_multiplier": 0.0},
            "lbp_alpha":   {"lr_multiplier": 1.0, "weight_decay_multiplier": 0.0},
            "rp_gates":    {"lr_multiplier": 3.0, "weight_decay_multiplier": 0.0},
            "classification_head": {"lr_multiplier": 0.7, "weight_decay_multiplier": 1.0}
        },

        # Optional gradient clipping
        "gradient_clipping": {
            "enabled": True,
            "method": "norm",
            "max_norm": 0.3
        },

        # Mixed precision and DataLoader settings
        "amp": True,
        "num_workers": 4,
        "pin_memory": True,

        # reproducibility / deterministic behavior
        "reproducibility": {
            "seed": 42,
            "deterministic": True,
            "benchmark": False
        },

        # "Hard fine-tune" phase settings (switch to hard discrete forward + bn recalibration)
        "hard_finetune": {
            "enable": True,
            "start_epoch": 190,
            "lr_mult": 0.1,
            "hard_forward_bits": True,
            "hard_forward_gates": True,
            "bn_mode": "track",
            "bn_recal_batches": 100
        },

        # rp_layer (RP gating/logit initialization)
        "rp_layer": {"gate_logits_init": 0.3},

        # Auxiliary regularizers to control sparsity / alive ratio of gates
        "alive_ratio_reg": {"enable": True, "target": 0.5, "weight": 1e-3},
        "gate_margin": {"enable": True, "margin": 1.0, "weight": 1e-3}
    },

    # New preset: Identical to paper_svhn_rp but for CROPPED SVHN (18x29). Only image_size differs.
    "paper_svhn_rp_cropped": {
        "model": "lbpnet_rp",
        "data": {"train_ratio": 0.85, "val_ratio": 0.15, "seed": 42},
        "image_size": [29, 18],  # (H, W) for cropped SVHN
        "lbp_layer": {
            "num_patterns": 2,
            "num_points": 8,
            "window": 5,
            "share_across_channels": True,
            "mode": "bits",
            "alpha_init": 0.2,
            "learn_alpha": True,
            "offset_init_std": 0.02
        },
        "blocks": {
            "stages": 8,
            "channels_per_stage": [37, 40, 80, 80, 160, 160, 320, 320],
            "downsample_at": [1, 3],
            "fusion_type": "rp_paper",
            "rp_config": {"fusion_type": "rp_paper", "n_bits_per_out": 8, "seed": 42, "threshold": None, "tau": 0.5, "use_ste": True}
        },
        "head": {"hidden": 512, "dropout_rate": 0.25, "num_classes": 10, "use_bn": True},
        "training": {
            "epochs": 220,
            "batch_size": 128,
            "lr": 1e-3,
            "weight_decay": 0.01,
            "lr_scheduler": "cosine",
            "warmup_epochs": 5,
            "patience": 80,
            "min_delta": 5e-4,
            "label_smoothing": 0.1
        },
        "ste": {"use_ste_bits": True, "use_ste_gates": True, "lbp_ste_scale": 6.0, "gate_ste_scale": 6.0},
        "optim": {"type": "AdamW", "lr": 1e-3, "weight_decay": 1e-2, "lr_mult": {"gates": 2.0, "offsets": 2.0, "base": 1.0}},
        "temp_schedule": {"alpha": {"start": 1.5, "end": 0.06, "mode": "cosine"}, "tau": {"start": 3.0, "end": 0.75, "mode": "cosine"}},
        "freeze_schedule": {"freeze_offsets_epochs": 5, "freeze_gates_extra_epochs": 2, "freeze_gate_epochs": 8},
        "stability": {"collapse_guard": True, "collapse_window": 5, "acc_drop_thr": 0.05, "alive_upper": 0.26},
        "ema": {"enable": True, "decay": 0.999},
        "augment": {"enable": False, "crop_pad": 2, "rotation": 10},
        "optimizer_groups": {
            "lbp_offsets": {"lr_multiplier": 5.0, "weight_decay_multiplier": 0.0},
            "lbp_alpha":   {"lr_multiplier": 1.0, "weight_decay_multiplier": 0.0},
            "rp_gates":    {"lr_multiplier": 3.0, "weight_decay_multiplier": 0.0},
            "classification_head": {"lr_multiplier": 0.7, "weight_decay_multiplier": 1.0}
        },
        "gradient_clipping": {"enabled": True, "method": "norm", "max_norm": 0.3},
        "amp": True, "num_workers": 4, "pin_memory": True,
        "reproducibility": {"seed": 42, "deterministic": True, "benchmark": False},
        "hard_finetune": {"enable": True, "start_epoch": 190, "lr_mult": 0.1, "hard_forward_bits": True, "hard_forward_gates": True, "bn_mode": "track", "bn_recal_batches": 100},
        "rp_layer": {"gate_logits_init": 0.3},
        "alive_ratio_reg": {"enable": True, "target": 0.5, "weight": 1e-3},
        "gate_margin": {"enable": True, "margin": 1.0, "weight": 1e-3}
    },

    "paper_mnist_rp_ft": {
        "model": "lbpnet_rp",
        "data": {"train_ratio": 0.85, "val_ratio": 0.15, "seed": 42},
        "image_size": 28,
        "lbp_layer": {
            "num_patterns": 2,
            "num_points": 8,
            "window": 3,
            "share_across_channels": True,
            "mode": "bits",
            "alpha_init": 0.2,
            "learn_alpha": True,
            "offset_init_std": 0.02
        },
        "blocks": {
            "stages": 3,
            "channels_per_stage": [39, 40, 80],
            "downsample_at": [1, 2],
            "fusion_type": "rp_paper",
            "rp_config": {"fusion_type": "rp_paper", "n_bits_per_out": 4, "seed": 42, "threshold": None, "tau": 3.0, "use_ste": True}
        },
        "head": {"hidden": 512, "dropout_rate": 0.0, "num_classes": 10, "use_bn": True},
        "training": {
            "epochs": 20,
            "batch_size": 128,
            "lr": 1e-3,
            "weight_decay": 0.01,
            "lr_scheduler": "cosine",
            "warmup_epochs": 0,
            "patience": 30,
            "min_delta": 5e-4,
            "label_smoothing": 0.1
        },
        "ste": {"use_ste_bits": True, "use_ste_gates": True, "lbp_ste_scale": 6.0, "gate_ste_scale": 6.0},
        "optim": {
            "type": "AdamW",
            "lr": 1e-3,
            "weight_decay": 1e-2,
            "lr_mult": {"gates": 2.0, "offsets": 2.0, "base": 1.0}
        },
        "temp_schedule": {
            "alpha": {"start": 1.5, "end": 0.08, "mode": "cosine"},
            "tau":   {"start": 3.0,  "end": 0.9,  "mode": "cosine"}
        },
        "freeze_schedule": {"freeze_offsets_epochs": 5, "freeze_gates_extra_epochs": 2, "freeze_gate_epochs": 8},
        "stability": {
            "collapse_guard": True,
            "collapse_window": 5,
            "acc_drop_thr": 0.05,
            "alive_upper": 0.26
        },
        "ema": {"enable": True, "decay": 0.999},
        "augment": {"enable": True, "crop_pad": 2, "rotation": 10},
        "optimizer_groups": {
            "lbp_offsets": {"lr_multiplier": 5.0, "weight_decay_multiplier": 0.0},
            "lbp_alpha":   {"lr_multiplier": 1.0, "weight_decay_multiplier": 0.0},
            "rp_gates":    {"lr_multiplier": 3.0, "weight_decay_multiplier": 0.0},
            "classification_head": {"lr_multiplier": 0.7, "weight_decay_multiplier": 1.0}
        },
        "gradient_clipping": {"enabled": True, "method": "norm", "max_norm": 0.3},
        "amp": True, "num_workers": 4, "pin_memory": True,
        "reproducibility": {"seed": 42, "deterministic": True, "benchmark": False},
        "hard_finetune": {
            "enable": True,
            "start_epoch": 0,
            "lr_mult": 0.1,
            "hard_forward_bits": True,
            "hard_forward_gates": True,
            "bn_mode": "track",
            "bn_recal_batches": 100
        },
        "rp_layer": {"gate_logits_init": 0.3},
        "alive_ratio_reg": {"enable": True, "target": 0.5, "weight": 1e-3},
        "gate_margin": {"enable": True, "margin": 1.0, "weight": 1e-3}
    },
    "paper_mnist_1x1": {
        "model": "lbpnet_rp",
        "data": {"train_ratio": 0.85, "val_ratio": 0.15, "seed": 42},
        "image_size": 28,
        "lbp_layer": {
            "num_patterns": 2, "num_points": 8, "window": 3,
            "share_across_channels": True, "mode": "bits",
            "alpha_init": 0.2, "learn_alpha": True, "offset_init_std": 0.8
        },
        "blocks": {
            "stages": 3,
            "channels_per_stage": [39, 40, 80],
            "downsample_at": [1, 2],
            "fusion_type": "conv1x1",
            "rp_config": {"fusion_type": "conv1x1", "n_bits_per_out": 4, "seed": 42}
        },
        "head": {"hidden": 512, "dropout_rate": 0.0, "num_classes": 10, "use_bn": True},
        "training": {
            "epochs": 220, "batch_size": 128, "lr": 1e-3, "weight_decay": 0.01,
            "lr_scheduler": "cosine", "warmup_epochs": 5, "patience": 80, "min_delta": 5e-4,
            "label_smoothing": 0.1
        },
        "ema": {"enable": True, "decay": 0.999},
        "augment": {"enable": True, "crop_pad": 2, "rotation": 10},
        "ste": {"use_ste_bits": True, "use_ste_gates": True},
        "temp_schedule": {
            "alpha_start": 1.5, "alpha_end": 0.03, "alpha_mode": "exp",
            "tau_start": 3.0, "tau_end": 0.2, "tau_mode": "exp"
        },
        "freeze_schedule": {"freeze_offsets_epochs": 2, "freeze_gates_extra_epochs": 2},
        "optimizer_groups": {
            "lbp_offsets": {"lr_multiplier": 5.0, "weight_decay_multiplier": 0.0},
            "lbp_alpha":   {"lr_multiplier": 1.0, "weight_decay_multiplier": 0.0},
            "rp_gates":    {"lr_multiplier": 3.0, "weight_decay_multiplier": 0.0},
            "classification_head": {"lr_multiplier": 0.7, "weight_decay_multiplier": 1.0}
        },
        "gradient_clipping": {"enabled": True, "method": "norm", "max_norm": 0.3},
        "amp": True, "num_workers": 4, "pin_memory": True,
        "reproducibility": {"seed": 42, "deterministic": True, "benchmark": False}
    ,
        "hard_finetune": {
            "enable": True,
            "last_k_epochs": 30,
            "lr_scale": 0.1,
            "alpha_hard": 0.03,
            "tau_hard": 0.2,
            "bn_recal_batches": 100,
            "freeze_offsets_epochs": 5,
            "bn_momentum": 0.01
        },
        "gate_margin": {"enable": True, "margin": 1.0, "weight": 1e-3}
    }
    ,
    "paper_mnist_rp_full": {
        "model": "lbpnet_rp",
        "data": {"val_size": 5000, "split_seed": 42, "stratified": True, "num_workers": 4, "pin_memory": True},
        "image_size": 28,
        "lbp_layer": {
            "num_patterns": 2,
            "num_points": 8,
            "window": 3,
            "share_across_channels": True,
            "mode": "bits",
            "alpha_init": 0.2,
            "learn_alpha": True,
            "offset_init_std": 0.02
        },
        "blocks": {
            "stages": 3,
            "channels_per_stage": [39, 40, 80],
            "downsample_at": [1, 2],
            "fusion_type": "rp_paper",
            "rp_config": {"fusion_type": "rp_paper", "n_bits_per_out": 4, "seed": 42, "threshold": None, "tau": 3.0, "use_ste": True}
        },
        "head": {"hidden": 256, "dropout_rate": 0.0, "num_classes": 10, "use_bn": True},
        "training": {
            "epochs": 300,
            "batch_size": 256,
            "lr": 1e-3,
            "weight_decay": 3e-4,
            "lr_scheduler": "cosine",
            "warmup_epochs": 5,
            "patience": 9999,
            "min_delta": 5e-4,
            "label_smoothing": 0.05
        },
        "ste": {"use_ste_bits": True, "use_ste_gates": True, "lbp_ste_scale": 6.0, "gate_ste_scale": 6.0},
        "optim": {
            "type": "AdamW",
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "lr_mult": {"gates": 2.0, "offsets": 2.0, "base": 1.0}
        },
        "temp_schedule": {
            "alpha": {"start": 1.5, "end": 0.08, "mode": "cosine"},
            "tau":   {"start": 3.0,  "end": 0.85, "mode": "cosine"}
        },
        "freeze_schedule": {"freeze_offsets_epochs": 8, "freeze_gates_extra_epochs": 4, "freeze_gate_epochs": 10},
        "stability": {
            "collapse_guard": True,
            "collapse_window": 5,
            "acc_drop_thr": 0.05,
            "alive_upper": 0.26
        },
        "ema": {"enable": True, "decay": 0.999},
        "augment": {"enable": True, "crop_pad": 2, "rotation": 10},
        "optimizer_groups": {
            "lbp_offsets": {"lr_multiplier": 5.0, "weight_decay_multiplier": 0.0},
            "lbp_alpha":   {"lr_multiplier": 1.0, "weight_decay_multiplier": 0.0},
            "rp_gates":    {"lr_multiplier": 3.0, "weight_decay_multiplier": 0.0},
            "classification_head": {"lr_multiplier": 0.7, "weight_decay_multiplier": 1.0}
        },
        "gradient_clipping": {"enabled": True, "method": "norm", "max_norm": 1.0},
        "amp": True, "num_workers": 4, "pin_memory": True,
        "reproducibility": {"seed": 42, "deterministic": True, "benchmark": False},
        "hard_finetune": {
            "enable": True,
            "start_epoch": 260,
            "lr_mult": 0.1,
            "hard_forward_bits": True,
            "hard_forward_gates": True,
            "bn_mode": "track",
            "bn_recal_batches": 200
        },
        "rp_layer": {"gate_logits_init": 0.3},
        "alive_ratio_reg": {"enable": True, "target": 0.5, "weight": 1e-3},
        "gate_margin": {"enable": True, "margin": 1.0, "weight": 1e-3}
    }
    ,
    "paper_mnist_rp_full_final": {
        "model": "lbpnet_rp",
        "data": {"val_size": 0, "split_seed": 42, "stratified": True, "num_workers": 4, "pin_memory": True},
        "image_size": 28,
        "lbp_layer": {
            "num_patterns": 2,
            "num_points": 8,
            "window": 3,
            "share_across_channels": True,
            "mode": "bits",
            "alpha_init": 0.2,
            "learn_alpha": True,
            "offset_init_std": 0.02
        },
        "blocks": {
            "stages": 3,
            "channels_per_stage": [39, 40, 80],
            "downsample_at": [1, 2],
            "fusion_type": "rp_paper",
            "rp_config": {"fusion_type": "rp_paper", "n_bits_per_out": 8, "seed": 42, "threshold": None, "tau": 3.0, "use_ste": True}
        },
        "head": {"hidden": 256, "dropout_rate": 0.0, "num_classes": 10, "use_bn": True},
        "training": {
            "epochs": 1,
            "batch_size": 256,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "lr_scheduler": "cosine",
            "warmup_epochs": 0,
            "patience": 1,
            "min_delta": 5e-4,
            "label_smoothing": 0.0
        },
        "ste": {"use_ste_bits": True, "use_ste_gates": True, "lbp_ste_scale": 6.0, "gate_ste_scale": 6.0},
        "optim": {"type": "AdamW", "lr": 1e-3, "weight_decay": 1e-4, "lr_mult": {"gates": 2.0, "offsets": 2.0, "base": 1.0}},
        "temp_schedule": {"alpha": {"start": 1.5, "end": 0.10, "mode": "cosine"}, "tau": {"start": 3.0, "end": 0.90, "mode": "cosine"}},
        "freeze_schedule": {"freeze_offsets_epochs": 0, "freeze_gates_extra_epochs": 0, "freeze_gate_epochs": 0},
        "stability": {"collapse_guard": False},
        "ema": {"enable": True, "decay": 0.999},
        "augment": {"enable": False},
        "optimizer_groups": {"lbp_offsets": {"lr_multiplier": 1.0, "weight_decay_multiplier": 0.0}, "lbp_alpha": {"lr_multiplier": 1.0, "weight_decay_multiplier": 0.0}, "rp_gates": {"lr_multiplier": 1.0, "weight_decay_multiplier": 0.0}, "classification_head": {"lr_multiplier": 1.0, "weight_decay_multiplier": 1.0}},
        "gradient_clipping": {"enabled": False, "method": "norm", "max_norm": 1.0},
        "amp": True, "num_workers": 4, "pin_memory": True,
        "reproducibility": {"seed": 42, "deterministic": True, "benchmark": False},
        "hard_finetune": {"enable": False},
        "rp_layer": {"gate_logits_init": 0.3}
    }
    ,
    "paper_mnist_rp_full60k": {
        "model": "lbpnet_rp",
        "data": {"val_size": 0, "split_seed": 42, "stratified": True, "num_workers": 4, "pin_memory": True},
        "image_size": 28,
        "lbp_layer": {
            "num_patterns": 2,
            "num_points": 8,
            "window": 3,
            "share_across_channels": True,
            "mode": "bits",
            "alpha_init": 0.2,
            "learn_alpha": True,
            "offset_init_std": 0.2
        },
        "blocks": {
            "stages": 3,
            "channels_per_stage": [39, 40, 80],
            "downsample_at": [1, 2],
            "fusion_type": "rp_paper",
            "rp_config": {"fusion_type": "rp_paper", "n_bits_per_out": 8, "seed": 42, "threshold": None, "tau": 3.0, "use_ste": True}
        },
        "head": {"hidden": 256, "dropout_rate": 0.0, "num_classes": 10, "use_bn": True},
        "training": {
            "epochs": 260,
            "batch_size": 256,
            "lr": 1e-3,
            "weight_decay": 3e-4,
            "lr_scheduler": "cosine",
            "warmup_epochs": 5,
            "patience": 9999,
            "min_delta": 5e-4,
            "label_smoothing": 0.03,
            "eta_min": 1e-6
        },
        "ste": {"use_ste_bits": True, "use_ste_gates": True, "lbp_ste_scale": 6.0, "gate_ste_scale": 8.0},
        "optim": {
            "type": "AdamW",
            "lr": 1e-3,
            "weight_decay": 2e-4,
            "lr_mult": {"gates": 2.0, "offsets": 2.0, "base": 1.0}
        },
        "temp_schedule": {
            "alpha": {"start": 1.5, "end": 0.08, "mode": "cosine"},
            "tau":   {"start": 3.0,  "end": 0.85, "mode": "cosine"}
        },
        "freeze_schedule": {"freeze_offsets_epochs": 8, "freeze_gates_extra_epochs": 4, "freeze_gate_epochs": 10},
        "stability": {"collapse_guard": True, "collapse_window": 5, "acc_drop_thr": 0.05, "alive_upper": 0.35},
        "ema": {"enable": True, "decay": 0.999},
        "augment": {"enable": True, "crop_pad": 2, "rotation": 2},
        "optimizer_groups": {
            "lbp_offsets": {"lr_multiplier": 5.0, "weight_decay_multiplier": 0.0},
            "lbp_alpha":   {"lr_multiplier": 1.0, "weight_decay_multiplier": 0.0},
            "rp_gates":    {"lr_multiplier": 3.0, "weight_decay_multiplier": 0.0},
            "classification_head": {"lr_multiplier": 0.7, "weight_decay_multiplier": 1.0}
        },
        "gradient_clipping": {"enabled": True, "method": "norm", "max_norm": 1.0},
        "amp": True, "num_workers": 4, "pin_memory": True,
        "reproducibility": {"seed": 42, "deterministic": True, "benchmark": False},
        "hard_finetune": {"enable": True, "start_epoch": 220, "lr_mult": 0.1, "hard_forward_bits": True, "hard_forward_gates": True, "bn_mode": "track", "bn_recal_batches": 400},
        "rp_layer": {"gate_logits_init": 0.3},
        "alive_ratio_reg": {"enable": True, "target": 0.5, "weight": 1e-3},
        "gate_margin": {"enable": True, "margin": 1.0, "weight": 1e-3}
    }
}

def get_config():
    """Prefer YAML via MODEL_CONFIG; otherwise use MODEL_PRESET. Apply env overrides as needed."""
    cfg = None
    cfg_src_label = None
    yaml_path = os.environ.get("MODEL_CONFIG", "").strip()
    if yaml_path and os.path.isfile(yaml_path):
        with open(yaml_path, 'r') as f:
            cfg = yaml.safe_load(f)
        cfg_src_label = f"yaml:{yaml_path}"
    else:
        # Default preset changed to SVHN-friendly preset
        preset = os.environ.get("MODEL_PRESET", "paper_svhn_rp")
        cfg_src = PRESETS.get(preset)
        if cfg_src is None:
            raise ValueError(f"Unknown MODEL_PRESET={preset}")
        cfg = copy.deepcopy(cfg_src)
        cfg_src_label = f"preset:{preset}"

    # Apply environment overrides (seed and RP bit width)
    try:
        env_seed = os.environ.get("MODEL_SEED", "").strip()
        if env_seed:
            seed_int = int(env_seed)
            if 'reproducibility' in cfg:
                cfg['reproducibility']['seed'] = seed_int
            if 'blocks' in cfg and 'rp_config' in cfg['blocks']:
                cfg['blocks']['rp_config']['seed'] = seed_int
        env_bits = os.environ.get("RP_BITS", "").strip()
        if env_bits:
            bits_int = int(env_bits)
            if 'blocks' in cfg and 'rp_config' in cfg['blocks']:
                cfg['blocks']['rp_config']['n_bits_per_out'] = bits_int
    except Exception:
        pass

    # Attach source label for logging/debugging
    cfg['_source'] = cfg_src_label

    # Auto-align data variant with chosen preset if not explicitly set
    # - If preset name contains 'cropped' → use cropped pipeline
    # - Otherwise default to full pipeline
    try:
        src = str(cfg.get('_source', ''))
        preset_name = src.split(':', 1)[1] if src.startswith('preset:') and (':' in src) else ''
        default_variant = 'cropped' if ('cropped' in preset_name.lower()) else 'full'
        data_cfg = cfg.setdefault('data', {})
        data_cfg.setdefault('variant', default_variant)
    except Exception:
        # If anything goes wrong, leave as-is; lbpnet.data defaults to 'cropped'
        pass
    return cfg


def estimate_gops(config, image_size=(28, 28)) -> float:
    """
    Estimate #Ops (GOPs). Convention:
    - LBP counts comparisons
    - RP counts n_bits_out per pixel
    - 1x1 convolution counts MACs
    image_size can be an int (square) or a (H, W) tuple.
    """
    if isinstance(image_size, (list, tuple)) and len(image_size) == 2:
        H, W = int(image_size[0]), int(image_size[1])
    else:
        H = W = int(image_size)

    blocks = config['blocks']
    lbp_cfg = config['lbp_layer']
    C_outs = blocks['channels_per_stage']
    fusion_type = blocks.get('fusion_type', 'rp')
    n_bits_out = blocks.get('rp_config', {}).get('n_bits_per_out', lbp_cfg['num_points'])
    down_at = set(blocks.get('downsample_at', []))

    total_ops = 0.0
    for i, C_out in enumerate(C_outs):
        # LBP comparisons per pixel
        lbp_ops = H * W * lbp_cfg['num_points']
        total_ops += lbp_ops
        # Fusion
        if fusion_type == 'conv1x1':
            C_in_bits = lbp_cfg['num_patterns'] * lbp_cfg['num_points']
            total_ops += H * W * C_in_bits * C_out
        else:
            total_ops += H * W * n_bits_out
        # Downsample resolution after this stage?
        if i in down_at:
            H = (H + 1) // 2
            W = (W + 1) // 2
    return total_ops / 1e9


def train_original_model():
    print("🚀 Starting training LBPNet with the original LBP layer on SVHN...")

    # Set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")

    # Training config - paper style preset
    config = get_config()

    # Announce which dataset variant/preset and input size we'll train on
    try:
        src = str(config.get('_source', 'preset:unknown'))
        variant = str(config.get('data', {}).get('variant', 'cropped'))
        H, W = get_image_hw(config)
        print(f"📦 Dataset variant: {variant} | preset={src} | expected input: {H}x{W}")
    except Exception as _e:
        print(f"📦 Dataset variant: <unknown> (info error: {_e})")

    # Set random seed
    torch.manual_seed(config['reproducibility']['seed'])
    np.random.seed(config['reproducibility']['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['reproducibility']['seed'])
        torch.cuda.manual_seed_all(config['reproducibility']['seed'])
        if config['reproducibility']['deterministic']:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = config['reproducibility']['benchmark']

    # Create SVHN datasets
    print("📊 Creating SVHN datasets...")
    train_dataset, val_dataset, test_dataset = get_svhn_datasets(config)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory']
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=config['num_workers'],
            pin_memory=config['pin_memory']
        )

    # Create original model
    print("🤖 Creating the original model...")
    model = build_model(config).to(device)
    # Optional: load trained weights for diagnostics (grid test/BN recalibration)
    ckpt_env = os.environ.get('LBP_LOAD_CKPT', '').strip()
    ckpt_default = os.path.join('./outputs_svhn_original', 'best_model.pth')
    # Support explicit disabling via env var values: '', 'NONE', 'NO'
    if ckpt_env == '' or ckpt_env.upper() in ('NONE', 'NO'):
        ckpt_path = ''
    else:
        # If env var supplied and points to a file, use it; otherwise fall back to default if present
        if ckpt_env and os.path.exists(ckpt_env):
            ckpt_path = ckpt_env
        else:
            ckpt_path = ckpt_default if os.path.exists(ckpt_default) else ''
    if ckpt_path:
        try:
            # Dummy forward to initialize runtime shapes (including RP mappings)
            # Note: switch to eval() to avoid BN issues with batch=1
            prev_train = model.training
            model.eval()
            H, W = get_image_hw(config)
            with torch.no_grad():
                _ = model(torch.zeros(8, 1, H, W, device=device))
            ckpt = torch.load(ckpt_path, map_location=device)
            raw_state = ckpt.get('model_state_dict', ckpt)
            incompatible = model.load_state_dict(raw_state, strict=False)
            print(f"📥 Loaded checkpoint (filtered rp_weights): {ckpt_path}\n   missing={len(incompatible.missing_keys)}, unexpected={len(incompatible.unexpected_keys)}")
            # Restore previous train/eval state
            if prev_train:
                model.train()
            else:
                model.eval()
        except Exception as e:
            print(f"⚠️ Failed to load weights: {ckpt_path}, err={e}")
    # Optionally zero classifier head bias
    if hasattr(model, 'fc_layers'):
        for m in model.fc_layers.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    # Print model info
    model_info = model.get_model_info()
    print(f"📋 Model info:")
    print(f"   Total parameters: {model_info['total_parameters']:,}")
    print(f"   LBP config: {model_info['lbp_config']}")
    print(f"   Block config: {model_info['block_config']}")
    # PAPER accounting for Size/GOPs (LBP + Fusion only)
    H, W = get_image_hw(config)
    paper_size = get_paper_model_size_bytes(model)
    paper_ops = estimate_ops_paper(model, (1,1,H, W))
    print(f"[PAPER] size_bytes={paper_size} ({paper_size/1024:.2f} KB), gops={paper_ops['gops_total']:.6f}, ops(cmp/add/mul)={paper_ops['cmps']}/{paper_ops['adds']}/{paper_ops['muls']}")

    # Create optimizer (param groups: gates/offsets ×2 LR)
    print("⚡ Creating optimizer...")
    base_lr = config.get('optim', {}).get('lr', config['training']['lr'])
    wd = config.get('optim', {}).get('weight_decay', config['training']['weight_decay'])
    lr_mult = config.get('optim', {}).get('lr_mult', {"gates": 2.0, "offsets": 2.0, "base": 1.0})

    gate_params = []
    offset_params = []
    if hasattr(model, 'collect_gate_params'):
        gate_params = list(model.collect_gate_params())
    if hasattr(model, 'collect_offset_params'):
        offset_params = list(model.collect_offset_params())

    all_params = [p for p in model.parameters() if p.requires_grad]
    gate_set = set(map(id, gate_params))
    offset_set = set(map(id, offset_params))
    base_params = [p for p in all_params if id(p) not in gate_set and id(p) not in offset_set]

    param_groups = []
    if base_params:
        param_groups.append({"params": base_params, "lr": base_lr * lr_mult.get('base', 1.0), "weight_decay": wd, "name": "base"})
    if gate_params:
        param_groups.append({"params": gate_params, "lr": base_lr * lr_mult.get('gates', 2.0), "weight_decay": wd, "name": "gates"})
    if offset_params:
        param_groups.append({"params": offset_params, "lr": base_lr * lr_mult.get('offsets', 2.0), "weight_decay": wd, "name": "offsets"})

    optim_type = config.get('optim', {}).get('type', 'AdamW').lower()
    if optim_type == 'adamw':
        optimizer = optim.AdamW(param_groups, lr=base_lr, weight_decay=wd, betas=(0.9, 0.999))
    else:
        optimizer = optim.Adam(param_groups, lr=base_lr, weight_decay=wd, betas=(0.9, 0.999))
    print(f"   Parameter group counts: base={len(base_params)}, gates={len(gate_params)}, offsets={len(offset_params)}")

    # LR scheduler
    if config['training']['lr_scheduler'] == 'cosine':
        eta_min = float(config['training'].get('eta_min', config['training']['lr'] * 0.01))
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['training']['epochs'],
            eta_min=eta_min
        )
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # Loss (with label smoothing)
    ls = float(config['training'].get('label_smoothing', 0.1))
    criterion = nn.CrossEntropyLoss(label_smoothing=ls)

    # Mixed precision
    scaler = GradScaler('cuda') if (config['amp'] and device.type == 'cuda') else None

    # Temperature scheduler (alpha/tau) and EMA
    ts_cfg = config.get('temp_schedule', {})
    alpha_cfg = ts_cfg.get('alpha', {"start": 1.5, "end": 0.06, "mode": "cosine"})
    tau_cfg   = ts_cfg.get('tau',   {"start": 3.0,  "end": 0.75, "mode": "cosine"})
    temp_sched = TemperatureScheduler(
        total_epochs=config['training']['epochs'],
        alpha_start=float(alpha_cfg.get('start', 1.5)), alpha_min=float(alpha_cfg.get('end', 0.06)),
        tau_start=float(tau_cfg.get('start', 3.0)), tau_min=float(tau_cfg.get('end', 0.75)),
        mode=str(alpha_cfg.get('mode', 'cosine')),
        guard_freeze_alpha_epochs=0,
    )
    ema_cfg = config.get('ema', {"enable": True, "decay": 0.999})
    use_ema = bool(ema_cfg.get('enable', False))
    ema = ModelEMA(model, decay=float(ema_cfg.get('decay', 0.999))) if use_ema else None

    # === Diagnostics helpers ===
    def set_bn_eval(m: nn.Module):
        for mod in m.modules():
            if isinstance(mod, nn.BatchNorm2d):
                mod.eval()
    def set_bn_train(m: nn.Module):
        for mod in m.modules():
            if isinstance(mod, nn.BatchNorm2d):
                mod.train()

    @torch.no_grad()
    def eval_accuracy(loader, soft_bits: bool, soft_gates: bool, hard_path: bool) -> float:
        if hard_path:
            model.eval()
        else:
            model.train()
            set_bn_eval(model)  # Get soft outputs in train mode without updating BN
        if hasattr(model, 'set_ste'):
            model.set_ste(True, True)  # Unified hard forward + STE
        total, correct = 0, 0
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            with autocast(device_type='cuda', enabled=(config['amp'] and device.type=='cuda')):
                out = model(data)
            pred = out.argmax(1)
            total += target.size(0)
            correct += (pred == target).sum().item()
        return 100.0 * correct / max(1, total)

    @torch.no_grad()
    def bn_recalibrate_hard(loader, max_batches: int = 100):
        # Update BN stats using hard path
        model.train()
        set_bn_train(model)
        if hasattr(model, 'set_ste'):
            model.set_ste(True, True)  # hard LBP + hard Gate via STE
        seen = 0
        # Slow BN momentum to reduce abrupt jumps
        bn_old_moms = []
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                bn_old_moms.append((m, m.momentum))
                m.momentum = 0.01
        for data, _ in loader:
            data = data.to(device)
            _ = model(data)
            seen += 1
            if seen >= max_batches:
                break
        for m, mom in bn_old_moms:
            m.momentum = mom
        model.eval()

    # 2x2 grid test trigger: A/B/C/D
    if os.environ.get('LBP_GRID_TEST', '0') == '1':
        print('🧪 2x2 soft/hard grid test (validation only):')
        model.eval()
        # construct validation loader
        val_loader = DataLoader(
            get_svhn_datasets(config)[1],
            batch_size=config['training']['batch_size'], shuffle=False,
            num_workers=config['num_workers'], pin_memory=config['pin_memory']
        )
        A = eval_accuracy(val_loader, soft_bits=True,  soft_gates=True,  hard_path=False)
        B = eval_accuracy(val_loader, soft_bits=False, soft_gates=True,  hard_path=False)
        C = eval_accuracy(val_loader, soft_bits=True,  soft_gates=False, hard_path=False)
        D = eval_accuracy(val_loader, soft_bits=False, soft_gates=False, hard_path=True)
        print(f"A soft bits + soft gates: {A:.2f}%")
        print(f"B hard bits + soft gates: {B:.2f}%")
        print(f"C soft bits + hard gates: {C:.2f}%")
        print(f"D hard bits + hard gates: {D:.2f}% (production path)")
        return model, None

    # BN recalibration then exit
    if os.environ.get('LBP_BN_RECAL', '0') == '1':
        print('🔧 Recalibrating BN statistics for hard path...')
        # use first few batches from training set
        train_loader_small = DataLoader(
            get_svhn_datasets(config)[0],
            batch_size=config['training']['batch_size'], shuffle=True,
            num_workers=0, pin_memory=False
        )
        bn_recalibrate_hard(train_loader_small, max_batches=100)
        print('✅ BN recalibration complete')
        return model, None

    # Optional: single-batch overfit self-test (32 images)
    if os.environ.get('LBP_OVERFIT_SELFTEST', '0') == '1':
        print("🔬 Single-batch overfit self-test: fixed 32 images...")
        model.train()
        subset_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=0)
        data32, target32 = next(iter(subset_loader))
        data32, target32 = data32.to(device), target32.to(device)
        # Disable Dropout; BN to eval
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                m.p = 0.0
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        # Fix alpha, open gates strongly, small tau
        if hasattr(model, 'set_ste'):
            model.set_ste(True, False)  # hard bits via STE; gates soft contribution
        if hasattr(model, 'update_alpha'):
            model.update_alpha(0.2)
        with torch.no_grad():
            for s in model.stages:
                # Reduce residual scaling to avoid identity dominance
                if hasattr(s, 'res_scale'):
                    s.res_scale.data.fill_(0.7)
                if hasattr(s, 'rp_layer') and hasattr(s.rp_layer, 'gate_logits') and (s.rp_layer.gate_logits is not None):
                    s.rp_layer.gate_logits.data.fill_(3.0)
                if hasattr(s, 'update_tau'):
                    s.update_tau(0.5)
                elif hasattr(s, 'rp_layer') and hasattr(s.rp_layer, 'set_tau'):
                    s.rp_layer.set_tau(0.5)

        # Optional: make RP weights learnable and init near-identity
        if os.environ.get('LBP_SELFTEST_RP_LEARN', '0') == '1':
            # Trigger forward once to initialize rp_weights shapes
            with torch.no_grad():
                _ = model(data32)
            import math
            for s in model.stages:
                if hasattr(s, 'rp_layer') and hasattr(s.rp_layer, 'rp_weights') and s.rp_layer.rp_weights.numel() > 0:
                    W = s.rp_layer.rp_weights.detach().clone().to(device)
                    C_out, D = W.shape
                    if os.environ.get('LBP_SELFTEST_RP_UNIT', '0') == '1':
                        W.zero_()
                        for c in range(C_out):
                            W[c, c % D] = 1.0 / math.sqrt(D)
                    # Replace with learnable param
                    s.rp_layer.rp_weights = nn.Parameter(W, requires_grad=True)

        # Self-test optimizer: no weight decay, larger LR for head/offsets
        head_params, offset_params, other_params = [], [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if 'offsets_raw' in name:
                offset_params.append(p)
            elif ('fc_layers' in name) or ('head' in name):
                head_params.append(p)
            else:
                other_params.append(p)
        opt = optim.Adam([
            {'params': head_params, 'lr': 3e-3},
            {'params': offset_params, 'lr': 2e-3},
            {'params': other_params, 'lr': 1e-3},
        ], betas=(0.9, 0.99), weight_decay=0.0)
        # Force gates open for diagnosis
        with torch.no_grad():
            for s in model.stages:
                if hasattr(s, 'rp_layer') and hasattr(s.rp_layer, 'gate_logits') and (s.rp_layer.gate_logits is not None):
                    s.rp_layer.gate_logits.data.fill_(3.0)
                if hasattr(s, 'update_tau') or hasattr(s.rp_layer, 'set_tau'):
                    if hasattr(s, 'update_tau'):
                        s.update_tau(0.5)
                    elif hasattr(s.rp_layer, 'set_tau'):
                        s.rp_layer.set_tau(0.5)
        for step in range(600):
            opt.zero_grad()
            with autocast(device_type='cuda', enabled=(config['amp'] and device.type=='cuda')):
                out = model(data32)
                loss = criterion(out, target32) + model.get_offset_penalty()
            # Optional: gate margin regularizer
            if config.get('gate_margin', {}).get('enable', False):
                margin = float(config['gate_margin'].get('margin', 1.0))
                weight = float(config['gate_margin'].get('weight', 1e-3))
                margin_losses = []
                for s in model.stages:
                    # Only if fusion layer has gate_logits
                    if hasattr(s, 'rp_layer') and hasattr(s.rp_layer, 'gate_logits') and (s.rp_layer.gate_logits is not None):
                        g = s.rp_layer.gate_logits
                        margin_losses.append(F.relu(margin - g.abs()).mean())
                if margin_losses:
                    loss = loss + weight * torch.stack(margin_losses).mean()

            if scaler:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward(); opt.step()
            with torch.no_grad():
                pred = out.argmax(1)
                acc = (pred==target32).float().mean().item()*100
                if (step+1)%50==0:
                    hist = torch.bincount(pred.detach().cpu(), minlength=config['head']['num_classes']).tolist()
                    print(f"  step {step+1:03d}: loss={loss.item():.4f}, acc={acc:.2f}%, pred_hist={hist}")
        print("✅ Self-test complete: if not >99%, please check the pipeline and hyperparameters.\n")
        # Self-test only
        if os.environ.get('LBP_SELFTEST_ONLY', '0') == '1':
            return model, None

    # Set STE switches (unified hard forward + STE) and inject ste_scale if supported
    if hasattr(model, 'set_ste'):
        model.set_ste(True, True)
    # Configure ste_scale if available
    ste_scales = config.get('ste', {})
    if ste_scales:
        # LBP ste_scale via alpha mapping: ste_scale = 1/alpha
        if hasattr(model, 'update_alpha') and ('lbp_ste_scale' in ste_scales):
            alpha_equiv = 1.0 / max(float(ste_scales.get('lbp_ste_scale', 6.0)), 1e-6)
            model.update_alpha(alpha_equiv)
        # RP gate ste_scale pass-through (if layer supports)
        for s in model.stages:
            if hasattr(s, 'fuse') and hasattr(s.fuse, 'set_gate_ste_scale'):
                s.fuse.set_gate_ste_scale(float(ste_scales.get('gate_ste_scale', 6.0)))

    # History
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    # NEW: histories for requested fields
    alpha_hist = []
    tau_hist = []
    alive_hard_hist = []
    guards_hist = []
    epoch_times_hist = []
    epoch_status_lines = []  # human-readable line per epoch (for epoch_status_log.txt)

    best_val_acc = 0.0
    patience_counter = 0

    # Output dir
    output_dir = "./outputs_svhn_original"
    os.makedirs(output_dir, exist_ok=True)

    # Save base LRs for warmup
    base_lrs = None

    # Training loop
    print("🎯 Starting training...")
    for epoch in range(config['training']['epochs']):
        epoch_start_time = time.time()

        # Anneal alpha and tau (tau -> gate_ste_scale)
        alpha, tau = temp_sched.step(epoch)
        if hasattr(model, 'update_alpha'):
            model.update_alpha(float(alpha))
        # tau to gate_ste_scale (larger tau -> softer -> smaller ste_scale)
        gate_ste_scale = 1.0 / max(float(tau), 1e-6)
        for s in getattr(model, 'stages', []):
            if hasattr(s, 'fuse') and hasattr(s.fuse, 'set_gate_ste_scale'):
                s.fuse.set_gate_ste_scale(gate_ste_scale)

        # Freeze/unfreeze offsets and gates
        freeze_offsets_epochs = config['freeze_schedule']['freeze_offsets_epochs']
        freeze_gates_extra = config['freeze_schedule']['freeze_gates_extra_epochs']
        for name, p in model.named_parameters():
            if 'offsets_raw' in name:
                p.requires_grad = epoch >= freeze_offsets_epochs
            if name.endswith('.alpha') or ('alpha' in name):
                p.requires_grad = epoch >= freeze_offsets_epochs
            if 'gate_logits' in name:
                p.requires_grad = epoch >= (freeze_offsets_epochs + freeze_gates_extra)

        # RP-guided bootstrapping (optional)
        if config.get('rp_bootstrap', {}).get('enable', False) and epoch == 0:
            # Init near-identity and make RP weights learnable
            with torch.no_grad():
                dummy_data = next(iter(train_loader))[0].to(device)
                _ = model(dummy_data)
            import math
            for s in model.stages:
                if hasattr(s, 'rp_layer') and hasattr(s.rp_layer, 'rp_weights') and s.rp_layer.rp_weights.numel() > 0:
                    W = s.rp_layer.rp_weights.detach().clone().to(device)
                    C_out, D = W.shape
                    W.zero_()
                    for c in range(C_out):
                        W[c, c % D] = 1.0 / math.sqrt(D)
                    s.rp_layer.rp_weights = nn.Parameter(W, requires_grad=True)
                if hasattr(s, 'rp_layer') and hasattr(s.rp_layer, 'gate_logits') and (s.rp_layer.gate_logits is not None):
                    with torch.no_grad():
                        s.rp_layer.gate_logits.data.fill_(3.0)
            # Smaller start tau
            if hasattr(model, 'update_tau'):
                model.update_tau(0.5)

        if config.get('rp_bootstrap', {}).get('enable', False) and epoch < config.get('rp_bootstrap', {}).get('unit_epochs', 0):
            if hasattr(model, 'update_tau'):
                tau_boot = 0.5 + 0.3 * (epoch / max(1, config['rp_bootstrap']['unit_epochs'] - 1))
                model.update_tau(tau_boot)
            if hasattr(model, 'set_ste'):
                model.set_ste(True, True)

        if config.get('rp_bootstrap', {}).get('enable', False) and epoch == config.get('rp_bootstrap', {}).get('unit_epochs', 0):
            with torch.no_grad():
                for s in model.stages:
                    if hasattr(s, 'rp_layer') and hasattr(s.rp_layer, 'gate_logits') and (s.rp_layer.gate_logits is not None):
                        s.rp_layer.gate_logits.data.fill_(-0.10)
            if hasattr(model, 'set_ste'):
                model.set_ste(config['ste']['use_ste_bits'], config['ste']['use_ste_gates'])

        # Freeze gates early
        model.freeze_gates(epoch < config.get('freeze_schedule', {}).get('freeze_gate_epochs', 0))

        # Hard fine-tune phase: hard forward + small LR + BN strategy
        hf = config.get('hard_finetune', {"enable": False})
        hard_phase = hf.get('enable', False) and (epoch >= hf.get('start_epoch', 10**9))
        if hard_phase:
            for pg in optimizer.param_groups:
                pg['lr'] = pg['lr'] * float(hf.get('lr_mult', 0.1))
            if hasattr(model, 'set_ste'):
                model.set_ste(use_ste_bits=hf.get('hard_forward_bits', True), use_ste_gates=hf.get('hard_forward_gates', True))
            set_bn_mode(model, hf.get('bn_mode', 'track'))
        else:
            if hasattr(model, 'set_ste'):
                model.set_ste(use_ste_bits=True, use_ste_gates=True)

        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        # Linear warmup
        if base_lrs is None:
            base_lrs = [g['lr'] for g in optimizer.param_groups]
        warmup_epochs = config['training'].get('warmup_epochs', 0)
        if epoch < warmup_epochs:
            warmup_factor = (epoch + 1) / max(1, warmup_epochs)
            for pg, base_lr in zip(optimizer.param_groups, base_lrs):
                pg['lr'] = base_lr * warmup_factor

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()

            if config['amp'] and scaler is not None:
                with autocast(device_type='cuda', enabled=(config['amp'] and device.type=='cuda')):
                    output = model(data)
                    loss = criterion(output, target)
                    # Offset regularizer
                    loss = loss + model.get_offset_penalty()
                    # Gate sparsity regularizer toward target alive ratio (optional)
                    if config.get('gate_sparsity', {}).get('enable', False):
                        target_alive = config['gate_sparsity'].get('target_alive', 0.5)
                        lambda_g = config['gate_sparsity'].get('weight', 1e-3)
                        gate_means = []
                        for s in model.stages:
                            if hasattr(s, 'get_gate_values'):
                                g = s.get_gate_values()
                                if g is not None:
                                    gate_means.append(g.mean())
                        if gate_means:
                            gate_mean = torch.stack(gate_means).mean()
                            loss = loss + lambda_g * F.l1_loss(gate_mean, torch.tensor(target_alive, device=gate_mean.device))
                    # Batch distribution regularizer (collapse mitigation)
                    with torch.no_grad():
                        p_batch = F.softmax(output, dim=1).mean(0)
                    uniform = torch.full_like(p_batch, 1.0 / config['head']['num_classes'])
                    dist_reg = F.kl_div((p_batch + 1e-8).log(), uniform, reduction='batchmean')
                    loss = loss + 1e-3 * dist_reg

                scaler.scale(loss).backward()

                if config['gradient_clipping']['enabled']:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clipping']['max_norm'])

                scaler.step(optimizer)
                scaler.update()
                if ema is not None:
                    ema.update(model)
            else:
                with autocast(device_type='cuda', enabled=False):
                    output = model(data)
                    loss = criterion(output, target)
                    loss = loss + model.get_offset_penalty()
                    if config.get('gate_sparsity', {}).get('enable', False):
                        target_alive = config['gate_sparsity'].get('target_alive', 0.5)
                        lambda_g = config['gate_sparsity'].get('weight', 1e-3)
                        gate_means = []
                        for s in model.stages:
                            if hasattr(s, 'get_gate_values'):
                                g = s.get_gate_values()
                                if g is not None:
                                    gate_means.append(g.mean())
                        if gate_means:
                            gate_mean = torch.stack(gate_means).mean()
                            loss = loss + lambda_g * F.l1_loss(gate_mean, torch.tensor(target_alive, device=gate_mean.device))
                    with torch.no_grad():
                        p_batch = F.softmax(output, dim=1).mean(0)
                    uniform = torch.full_like(p_batch, 1.0 / config['head']['num_classes'])
                    dist_reg = F.kl_div((p_batch + 1e-8).log(), uniform, reduction='batchmean')
                    loss = loss + 1e-3 * dist_reg  # <-- (fixed stray parentheses)

                loss.backward()

                if config['gradient_clipping']['enabled']:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clipping']['max_norm'])

                optimizer.step()
                if ema is not None:
                    ema.update(model)

            # Stats
            train_loss += loss.item()

            _, predicted = output.max(1)
            train_total += target.size(0)
            train_correct += predicted.eq(target).sum().item()

        # Training metrics
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total

        # Validation
        model.eval()
        if val_loader is not None:
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            pred_hist = torch.zeros(config['head']['num_classes'], dtype=torch.long)
            with torch.no_grad():
                # Evaluate with EMA weights if enabled
                ema_backup = None
                if 'use_ema' in locals() and use_ema and (ema is not None):
                    ema_backup = ema.apply_to(model)
                for data, target in val_loader:
                    data, target = data.to(device), target.to(device)
                    with autocast(device_type='cuda', enabled=(config['amp'] and device.type=='cuda')):
                        output = model(data)
                        loss = criterion(output, target) + model.get_offset_penalty()
                    val_loss += loss.item()
                    _, predicted = output.max(1)
                    val_total += target.size(0)
                    val_correct += predicted.eq(target).sum().item()
                    pred_hist += torch.bincount(predicted.detach().cpu(), minlength=config['head']['num_classes'])
                if ema_backup is not None:
                    ModelEMA.restore(model, ema_backup)
            # During hard phase: BN recalibration with training loader
            if hard_phase:
                bn_recalibrate_hard(train_loader, max_batches=int(hf.get('bn_recal_batches', 100)))
            # Validation metrics
            val_loss = val_loss / max(1, len(val_loader))
            val_acc = 100. * val_correct / max(1, val_total)
        else:
            # No validation set: mirror training metrics
            val_loss = train_loss
            val_acc = train_acc
            pred_hist = torch.zeros(config['head']['num_classes'], dtype=torch.long)

        # Step LR
        scheduler.step()

        # Collapse guard: if two consecutive significant drops and high alive ratio -> soften and restore from best
        if config.get('stability', {}).get('collapse_guard', True):
            if len(val_accs) >= 2:
                drop_thr = float(config['stability'].get('acc_drop_thr', 0.05))
                alive_upper = float(config['stability'].get('alive_upper', 0.26))
                recent_drop = (val_accs[-1] < (1.0 - drop_thr) * max(best_val_acc, 1e-6)) and (val_accs[-2] < (1.0 - drop_thr) * max(best_val_acc, 1e-6))
                est_alive = None
                gstats = model.gate_stats() if hasattr(model, 'gate_stats') else None
                if gstats:
                    est_alive = sum(s['alive_hard'] for s in gstats) / len(gstats)
                if recent_drop and (est_alive is not None) and (est_alive >= alive_upper):
                    temp_sched.soften(extra_freeze_alpha_steps=3)
                    # Restore from best checkpoint if present
                    best_model_path = os.path.join(output_dir, 'best_model.pth')
                    if os.path.exists(best_model_path):
                        try:
                            ckpt = torch.load(best_model_path, map_location=device)
                            model.load_state_dict(ckpt.get('model_state_dict', ckpt), strict=False)
                            # Sync EMA shadow
                            if ('ema_shadow' in ckpt) and (ema is not None) and ckpt['ema_shadow']:
                                ema.shadow = ckpt['ema_shadow']
                            print("   🛡️ Collapse guard: restored from best and softened tau/froze alpha briefly")
                        except Exception as e:
                            print(f"   ⚠️ Collapse guard restore failed: {e}")

        # Record history
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # Epoch summary
        epoch_time = time.time() - epoch_start_time

        # Alive ratio summary + requested fields
        alive_ratios = []
        if hasattr(model, 'stages'):
            for s in model.stages:
                if hasattr(s, 'get_alive_ratio'):
                    alive_ratios.append(s.get_alive_ratio())
        mean_alive = sum(alive_ratios)/len(alive_ratios) if alive_ratios else 0.0
        gstats = model.gate_stats() if hasattr(model, 'gate_stats') else None
        guard_events = getattr(temp_sched, 'guard_events', 0)
        if gstats:
            alive_hard = sum(s['alive_hard'] for s in gstats) / len(gstats)
            alive_soft = sum(s['alive_soft'] for s in gstats) / len(gstats)
            mean_soft  = sum(s['mean_soft']  for s in gstats) / len(gstats)
            std_soft   = sum(s['std_soft']   for s in gstats) / len(gstats)
            print(f"\n📊 Epoch {epoch+1}/{config['training']['epochs']} summary:")
            print(f"   Train loss: {train_loss:.4f}, Train accuracy: {train_acc:.2f}%")
            print(f"   Val loss: {val_loss:.4f}, Val accuracy: {val_acc:.2f}%")
            print(f"   Current Alpha: {alpha:.4f}, Tau: {tau:.4f}, Alive(hard): {alive_hard:.3f}, Alive(soft): {alive_soft:.3f}, gate(mu±sigma)={mean_soft:.3f}±{std_soft:.3f}, guards={guard_events}")
        else:
            alive_hard = mean_alive
            print(f"\n📊 Epoch {epoch+1}/{config['training']['epochs']} summary:")
            print(f"   Train loss: {train_loss:.4f}, Train accuracy: {train_acc:.2f}%")
            print(f"   Val loss: {val_loss:.4f}, Val accuracy: {val_acc:.2f}%")
            print(f"   Current Alpha: {alpha:.4f}, Tau: {tau:.4f}, Alive(hard): {alive_hard:.3f}, guards={guard_events}")

        print(f"   Validation prediction histogram: {[int(x) for x in pred_hist.tolist()]}")
        print(f"   Time elapsed: {epoch_time:.2f}s")

        # --- NEW: store requested fields for logs ---
        alpha_hist.append(float(alpha))
        tau_hist.append(float(tau))
        alive_hard_hist.append(float(alive_hard))
        guards_hist.append(int(guard_events))
        epoch_times_hist.append(float(epoch_time))
        epoch_status_lines.append(
            f"Epoch {epoch+1}: Current Alpha: {alpha:.4f}, Tau: {tau:.4f}, Alive(hard): {alive_hard:.3f}, guards={guard_events}, time={epoch_time:.2f}s"
        )

        # Collapse-guard update based on validation feedback
        if gstats:
            stab = config.get('stability', {})
            temp_sched.update_on_val(val_acc, alive_hard, acc_drop_thr=float(stab.get('acc_drop_thr', 0.05)), alive_upper=float(stab.get('alive_upper', 0.26)))

        # Check for best model
        if val_acc > best_val_acc + config['training']['min_delta']:
            best_val_acc = val_acc
            patience_counter = 0

            # Save best model
            best_model_path = os.path.join(output_dir, 'best_model.pth')
            ckpt_extra = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': (scaler.state_dict() if scaler is not None else None),
                'temp_sched': temp_sched.state_dict(),
                'ema_shadow': (ema.shadow if (ema is not None) else None),
                'best_val_acc': best_val_acc,
                'config': config
            }
            torch.save(ckpt_extra, best_model_path)
            print(f"   🎉 New best model saved: {best_model_path}")
        else:
            patience_counter += 1
            print(f"   ⏳ Patience count: {patience_counter}/{config['training']['patience']}")

        # Early stop
        if patience_counter >= config['training']['patience']:
            print(f"\n🛑 Early stopping triggered! No improvement for {config['training']['patience']} epochs")
            break

        print("-" * 80)

    # Training complete
    print(f"\n🎉 Training complete! Best validation accuracy: {best_val_acc:.2f}%")

    # Save final model
    final_model_path = os.path.join(output_dir, 'final_model.pth')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': (scaler.state_dict() if scaler is not None else None),
        'temp_sched': temp_sched.state_dict(),
        'ema_shadow': (ema.shadow if (ema is not None) else None),
        'best_val_acc': best_val_acc,
        'config': config
    }, final_model_path)
    print(f"📁 Final model saved: {final_model_path}")

    # ===== Save training logs and plots to output_dir =====
    try:
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, 'training_log.csv')
        txt_path = os.path.join(output_dir, 'training_log.txt')
        status_txt_path = os.path.join(output_dir, 'epoch_status_log.txt')
        loss_plot_path = os.path.join(output_dir, 'loss_plot.png')
        acc_plot_path = os.path.join(output_dir, 'acc_plot.png')

        # Save CSV log: epoch, train_loss, val_loss, train_acc, val_acc, alpha, tau, alive_hard, guards, epoch_time
        n = max(len(train_losses), len(val_losses), len(train_accs), len(val_accs))
        with open(csv_path, 'w', newline='') as fcsv:
            writer = csv.writer(fcsv)
            writer.writerow(['epoch', 'train_loss', 'val_loss', 'train_acc', 'val_acc',
                             'alpha', 'tau', 'alive_hard', 'guards', 'epoch_time_sec'])
            for i in range(n):
                row = [
                    i+1,
                    float(train_losses[i]) if i < len(train_losses) else '',
                    float(val_losses[i]) if i < len(val_losses) else '',
                    float(train_accs[i]) if i < len(train_accs) else '',
                    float(val_accs[i]) if i < len(val_accs) else '',
                    float(alpha_hist[i]) if i < len(alpha_hist) else '',
                    float(tau_hist[i]) if i < len(tau_hist) else '',
                    float(alive_hard_hist[i]) if i < len(alive_hard_hist) else '',
                    int(guards_hist[i]) if i < len(guards_hist) else '',
                    float(epoch_times_hist[i]) if i < len(epoch_times_hist) else '',
                ]
                writer.writerow(row)

        # Save a plain text log with epoch summaries (now includes the new fields)
        with open(txt_path, 'w') as ftxt:
            ftxt.write(f"Training log for model saved in: {output_dir}\n")
            ftxt.write(f"Best validation accuracy: {best_val_acc:.4f}\n")
            ftxt.write("epoch,train_loss,val_loss,train_acc,val_acc,alpha,tau,alive_hard,guards,epoch_time_sec\n")
            for i in range(n):
                ftxt.write(f"{i+1},"
                           f"{(train_losses[i] if i < len(train_losses) else '')},"
                           f"{(val_losses[i] if i < len(val_losses) else '')},"
                           f"{(train_accs[i] if i < len(train_accs) else '')},"
                           f"{(val_accs[i] if i < len(val_accs) else '')},"
                           f"{(alpha_hist[i] if i < len(alpha_hist) else '')},"
                           f"{(tau_hist[i] if i < len(tau_hist) else '')},"
                           f"{(alive_hard_hist[i] if i < len(alive_hard_hist) else '')},"
                           f"{(guards_hist[i] if i < len(guards_hist) else '')},"
                           f"{(epoch_times_hist[i] if i < len(epoch_times_hist) else '')}\n")

        # Save the exact human-readable lines as they appear in console (one per epoch)
        with open(status_txt_path, 'w') as fs:
            fs.write("Per-epoch status lines (alpha/tau/alive_hard/guards/time):\n")
            for line in epoch_status_lines:
                fs.write(line + "\n")

        # Save plots (loss and accuracy)
        try:
            epochs = list(range(1, n+1))
            # Loss
            plt.figure()
            if len(train_losses) > 0:
                plt.plot(epochs[:len(train_losses)], train_losses, label='train_loss')
            if len(val_losses) > 0:
                plt.plot(epochs[:len(val_losses)], val_losses, label='val_loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(loss_plot_path)
            plt.close()

            # Accuracy
            plt.figure()
            if len(train_accs) > 0:
                plt.plot(epochs[:len(train_accs)], train_accs, label='train_acc')
            if len(val_accs) > 0:
                plt.plot(epochs[:len(val_accs)], val_accs, label='val_acc')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy (%)')
            plt.title('Training and Validation Accuracy')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(acc_plot_path)
            plt.close()
        except Exception as e:
            print(f"⚠️ Failed to save plots: {e}")

        print(f"✅ Training logs and plots saved in: {output_dir}")
        print(f"   - CSV: {csv_path}")
        print(f"   - TXT (table): {txt_path}")
        print(f"   - Status lines: {status_txt_path}")
    except Exception as e:
        print(f"⚠️ Failed to write training logs/plots to {output_dir}: {e}")

    # ===== Final: load best -> BN recalibration (hard path) -> evaluate on official test =====
    try:
        best_model_path = os.path.join(output_dir, 'best_model.pth')
        if os.path.exists(best_model_path):
            print("===> Final evaluation: load best weights, hard-path BN recalibration, then TEST ...")
            ckpt = torch.load(best_model_path, map_location=device)
            raw_state = ckpt.get('model_state_dict', ckpt)
            # Dummy forward to initialize dynamic buffers
            prev_mode = model.training
            model.eval()
            with torch.no_grad():
                H, W = get_image_hw(config)
                _ = model(torch.zeros(8, 1, H, W, device=device))
            _incompat = model.load_state_dict(raw_state, strict=False)
            if hasattr(model, 'set_ste'):
                model.set_ste(True, True)
            # Use cropped training dataset as calibration source (matches input shape)
            calib_loader = DataLoader(
                train_dataset,
                batch_size=config['training']['batch_size'],
                shuffle=True, num_workers=0, pin_memory=False
            )
            bn_recalibrate_hard(calib_loader, max_batches=int(config.get('hard_finetune', {}).get('bn_recal_batches', 200)))
            # Evaluate on test set
            test_loss = 0.0
            test_correct = 0
            test_total = 0
            model.eval()
            with torch.no_grad():
                for data, target in DataLoader(test_dataset, batch_size=config['training']['batch_size'], shuffle=False, num_workers=0):
                    data, target = data.to(device), target.to(device)
                    with autocast(device_type='cuda', enabled=(config['amp'] and device.type=='cuda')):
                        output = model(data)
                        loss = criterion(output, target)
                    test_loss += float(loss.item())
                    pred = output.argmax(1)
                    test_total += int(target.size(0))
                    test_correct += int((pred == target).sum().item())
            test_loss = test_loss / max(1, len(test_dataset) // config['training']['batch_size'])
            test_acc = 100.0 * test_correct / max(1, test_total)
            print(f"[FINAL] test_acc={test_acc:.4f}%, test_loss={test_loss:.4f}")
            # Restore previous mode
            if prev_mode:
                model.train()
    except Exception as _e:
        print(f"[FINAL] Final evaluation failed: {_e}")

    return model, best_val_acc

if __name__ == "__main__":
    train_original_model()
