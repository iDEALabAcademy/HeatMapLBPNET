#!/usr/bin/env python3
"""
Train LBPNet with original LBP layer on AMNIST (cropped 19x18).
Updated: heatmap-biased offset initialization removed; offsets always Gaussian.
Adaptive-P logic retained (optional) for per-pixel P selection using window mean.
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
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import matplotlib
matplotlib.use('Agg')  # headless backend for servers/SSH
import matplotlib.pyplot as plt
import time
import yaml
import csv
import json  # <-- used for adaptive-P JSON summary (single file)
# For plotting window-mean distribution of the global heatmap
try:
    from lbpnet.data.heatmap import window_mean_from_heatmap
except Exception:
    window_mean_from_heatmap = None

from lbpnet.models import build_model
from tools.metrics_paper import get_paper_model_size_bytes, estimate_ops_paper
from lbpnet.data import get_ckplus_datasets

# Debug flag: disable ALL filesystem writes when False
SAVE_ENABLED = True

# -------------------------- small helpers --------------------------

def anneal(epoch, max_epoch, start, end, mode="exp"):
    if mode == "exp":
        r = (end / max(start, 1e-8)) ** (epoch / max(1, max_epoch))
        return start * r
    t = epoch / max(1, max_epoch)
    return start + (end - start) * t

def interp(v0, v1, t, mode="exp"):
    t = float(max(0.0, min(1.0, t)))
    if mode == "exp":
        v0 = max(1e-8, float(v0)); v1 = max(1e-8, float(v1))
        return v0 * ((v1 / v0) ** t)
    return v0 + (v1 - v0) * t

def get_image_hw(cfg) -> Tuple[int, int]:
    sz = cfg.get('image_size', 32)
    if isinstance(sz, (list, tuple)) and len(sz) == 2:
        return int(sz[0]), int(sz[1])
    return int(sz), int(sz)

# -------------------------- device selection --------------------------

def safe_select_device(prefer_idx: int = 1) -> torch.device:
    """Pick a usable CUDA device if available, otherwise CPU.
    Tries the preferred index first; on failure, falls back to the first available GPU; lastly CPU.
    """
    if torch.cuda.is_available():
        try:
            n = torch.cuda.device_count()
        except Exception:
            n = 0
        candidates = []
        if isinstance(prefer_idx, int) and prefer_idx is not None:
            candidates.append(prefer_idx)
        candidates += [i for i in range(n) if i != prefer_idx]
        for i in candidates:
            if i < 0 or i >= n:
                continue
            try:
                # Quick probe tensor on device i
                _ = torch.tensor(0, device=f'cuda:{i}')
                return torch.device(f'cuda:{i}')
            except Exception:
                continue
    return torch.device('cpu')

# -------------------------- schedulers/EMA --------------------------

class TemperatureScheduler:
    """Unified manager for alpha/tau annealing and collapse-guard softening."""
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
        self._tau_soften_multiplier = 1.0
        self._acc_window = deque(maxlen=5)

    def _cosine(self, start: float, end: float, t: float) -> float:
        import math
        t = max(0.0, min(1.0, t))
        return end + (start - end) * 0.5 * (1 + math.cos(math.pi * t))

    def step(self, epoch: Optional[int] = None) -> Tuple[float, float]:
        if epoch is not None: self._epoch = int(epoch)
        t = self._epoch / max(1, self.total_epochs - 1)
        if self.mode == "cosine":
            alpha = self._cosine(self.alpha_start, self.alpha_min, t)
            tau   = self._cosine(self.tau_start,  self.tau_min,  t)
        else:
            alpha = interp(self.alpha_start, self.alpha_min, t, mode="exp")
            tau   = interp(self.tau_start,  self.tau_min,  t, mode="exp")
        if self._freeze_alpha_steps > 0:
            alpha = max(alpha, self.alpha_min * 1.5)
            self._freeze_alpha_steps -= 1
        tau = min(self.tau_start, tau * self._tau_soften_multiplier)
        return float(alpha), float(tau)

    def update_on_val(self, val_acc: float, alive_hard: float,
                      acc_drop_thr: float = 0.05, alive_upper: float = 0.26) -> None:
        self._acc_window.append(float(val_acc))
        if len(self._acc_window) < self._acc_window.maxlen: return
        avg5 = sum(self._acc_window) / len(self._acc_window)
        if (alive_hard is not None and alive_hard > float(alive_upper)) or (self._acc_window[-1] < (1.0 - float(acc_drop_thr)) * avg5):
            self._tau_soften_multiplier = min(self._tau_soften_multiplier * 1.08, 2.0)
            self._freeze_alpha_steps = max(self._freeze_alpha_steps, 3)
            self._guard_events += 1

    def soften(self, extra_freeze_alpha_steps: int = 3, tau_cap: Optional[float] = None):
        self._tau_soften_multiplier = min(self._tau_soften_multiplier * 1.25, 3.0)
        self._freeze_alpha_steps = max(self._freeze_alpha_steps, int(extra_freeze_alpha_steps))
        if tau_cap is not None:
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
        if not state: return
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

# -------------------------- presets (HM only, for more configs see original file) --------------------------
# --- CK+ preset (cropped 37x37) ---
PRESETS = {"paper_ckplus_rp_cropped": {
    "model": "lbpnet_rp",

    # Use CK+ cropped 37x37 (from bbox)
    "data": { 
        "variant": "cropped",
        "train_ratio": 0.80, 
        "val_ratio": 0.10, 
        "seed": 42,
        "bbox_path": "/home/sgram/Heatmap/CK+/Box/avg_top40_CK+/bbox.npy",
        "normalize": True
    },
    "image_size": 37,  # CK+ cropped is square 37x37

    # LBP layer configuration with heatmap support
    "lbp_layer": {
        "num_patterns": 2,
        "num_points_stage0": 12,  # Pmax for adaptive stage0
        "num_points": 8,          # default for subsequent layers (overridden by per_stage)
        # Per-stage P decay: stage0 adaptive, stage1=6, stage2=4
        "num_points_per_stage": [12, 8, 8],
        "window": 5,
        "share_across_channels": True,
        "mode": "bits",
        "alpha_init": 0.2,
        "learn_alpha": True,
        "offset_init_std": 0.02,
        # Heatmap path for adaptive-P or visualization (NEW)
        "heatmap_path": "/home/sgram/Heatmap/CK+/Box/avg_top40_CK+/cropped_heatmaps/heatmap_crop.npy",

        # >>> Adaptive-P configuration (NEW) - layer 1 only <<<
        "adaptive_p_enable": True,
        "adaptive_p_apply_only_stage0": True,
        "adaptive_p_thresholds": [0.35, 0.50, 0.70, 0.85],
        "adaptive_p_values": [4, 6, 8, 10, 12],
        "adaptive_perm_seed": 42
    },

    "blocks": {
        "stages": 3,
        "channels_per_stage": [39, 40, 80],
        "downsample_at": [1, 2],
        "fusion_type": "rp_paper",
        "rp_config": {
            "fusion_type": "rp_paper",
            "n_bits_per_out": 8,
            "seed": 42,
            "threshold": None,
            "tau": 0.5,
            "use_ste": True
        }
    },

    "head": { "hidden": 512, "dropout_rate": 0.25, "num_classes": 7, "use_bn": True },

    "training": {
        "epochs": 500,
        "batch_size": 16,
        "lr": 5e-4,
        "weight_decay": 1e-4,
        "lr_scheduler": "cosine",
        "warmup_epochs": 10,
        "patience": 150,
        "min_delta": 5e-4,
        "label_smoothing": 0.05,
        "eta_min": 1e-6
    },

    "ste": { "use_ste_bits": True, "use_ste_gates": True, "lbp_ste_scale": 6.0, "gate_ste_scale": 6.0 },

    "optim": { "type": "AdamW", "lr": 5e-4, "weight_decay": 1e-4,
               "lr_mult": { "gates": 2.0, "offsets": 2.0, "base": 1.0 } },

    "temp_schedule": {
        "alpha": { "start": 1.5, "end": 0.06, "mode": "cosine" },
        "tau":   { "start": 3.0, "end": 0.75, "mode": "cosine" }
    },

    "freeze_schedule": { "freeze_offsets_epochs": 0, "freeze_gates_extra_epochs": 0, "freeze_gate_epochs": 0 },

    "stability": { "collapse_guard": True, "collapse_window": 5, "acc_drop_thr": 0.05, "alive_upper": 0.26 },

    "ema": { "enable": True, "decay": 0.999 },

    "augment": { "enable": False },

    "optimizer_groups": {
        "lbp_offsets": { "lr_multiplier": 5.0, "weight_decay_multiplier": 0.0 },
        "lbp_alpha":   { "lr_multiplier": 1.0, "weight_decay_multiplier": 0.0 },
        "rp_gates":    { "lr_multiplier": 3.0, "weight_decay_multiplier": 0.0 },
        "classification_head": { "lr_multiplier": 1.5, "weight_decay_multiplier": 1.0 }
    },

    "gradient_clipping": { "enabled": True, "method": "norm", "max_norm": 1.0 },

    "amp": True, "num_workers": 4, "pin_memory": True,

    "reproducibility": { "seed": 42, "deterministic": True, "benchmark": False },

    "hard_finetune": {
        "enable": True,
        "start_epoch": 190,
        "lr_mult": 0.1,
        "hard_forward_bits": True,
        "hard_forward_gates": True,
        "bn_mode": "track",
        "bn_recal_batches": 100
    },

    "rp_layer": { "gate_logits_init": 0.3 },

    "alive_ratio_reg": { "enable": True, "target": 0.5, "weight": 1e-3 },
    "gate_margin": { "enable": True, "margin": 1.0, "weight": 1e-3 }
} 
}

# -------------------------- config loader + heatmap overrides --------------------------

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
        preset = os.environ.get("MODEL_PRESET", "paper_ckplus_rp_cropped")
        from_this_module = globals().get('PRESETS', {})
        cfg_src = from_this_module.get(preset)
        if cfg_src is None:
            raise ValueError(f"Unknown MODEL_PRESET={preset}")
        cfg = copy.deepcopy(cfg_src)
        cfg_src_label = f"preset:{preset}"

    # seed & RP bits env overrides
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

    # Auto variant
    try:
        src = str(cfg.get('_source', ''))
        preset_name = src.split(':', 1)[1] if src.startswith('preset:') and (':' in src) else ''
        default_variant = 'cropped' if ('cropped' in preset_name.lower()) else 'full'
        data_cfg = cfg.setdefault('data', {})
        data_cfg.setdefault('variant', default_variant)
    except Exception:
        pass

    # (Removed) Heatmap-biased init env overrides. Offsets always use Gaussian init.

    return cfg

# -------------------------- GOPs estimator (unchanged) --------------------------

def estimate_gops(config, image_size=(28, 28)) -> float:
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
    # Per-stage P override (if provided)
    per_stage_P = lbp_cfg.get('num_points_per_stage', None)
    total_ops = 0.0
    for i, C_out in enumerate(C_outs):
        P_i = int(per_stage_P[i]) if (per_stage_P and i < len(per_stage_P)) else int(lbp_cfg['num_points'])
        lbp_ops = H * W * P_i
        total_ops += lbp_ops
        if fusion_type == 'conv1x1':
            C_in_bits = lbp_cfg['num_patterns'] * P_i
            total_ops += H * W * C_in_bits * C_out
        else:
            total_ops += H * W * n_bits_out
        if i in down_at:
            H = (H + 1) // 2
            W = (W + 1) // 2
    return total_ops / 1e9

# -------------------------- main train loop (mostly unchanged) --------------------------

def train_original_model():
    print("🚀 Starting training LBPNet with the original LBP layer on CK+ cropped (37x37)...")

    device = safe_select_device(prefer_idx=1)
    print(f"🔧 Using device: {device}")

    config = get_config()

    # Log dataset/preset/heatmap status
    try:
        src = str(config.get('_source', 'preset:unknown'))
        variant = str(config.get('data', {}).get('variant', 'cropped'))
        H, W = get_image_hw(config)
        print(f"📦 Dataset variant: {variant} | preset={src} | expected input: {H}x{W}")
    except Exception as _e:
        print(f"📦 Dataset variant: <unknown> (info error: {_e})")

    # Seeds / cudnn
    torch.manual_seed(config['reproducibility']['seed'])
    np.random.seed(config['reproducibility']['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['reproducibility']['seed'])
        torch.cuda.manual_seed_all(config['reproducibility']['seed'])
        # Optional speed-tune (SPEED_TUNE=1) enables cuDNN autotune and disables determinism
        speed_tune = os.environ.get('SPEED_TUNE', '').strip().lower() in ('1', 'true', 'yes')
        if config['reproducibility']['deterministic'] and not speed_tune:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = config['reproducibility']['benchmark']
        else:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True

    # Datasets / loaders
    print("📊 Creating CK+ cropped datasets...")
    train_dataset, val_dataset, test_dataset = get_ckplus_datasets(config)
    # Runtime speed knobs via env (defaults keep current behavior)
    validate_every = max(1, int(os.environ.get('VALIDATE_EVERY', '1') or '1'))
    prefetch_factor = int(os.environ.get('PREFETCH_FACTOR', '4') or '4')
    persistent_workers = (os.environ.get('PERSISTENT_WORKERS', '1') not in ('0', 'false', 'False'))

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory'],
        persistent_workers=(persistent_workers if config['num_workers'] > 0 else False),
        prefetch_factor=(prefetch_factor if config['num_workers'] > 0 else None),
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=config['num_workers'],
            pin_memory=config['pin_memory'],
            persistent_workers=(persistent_workers if config['num_workers'] > 0 else False),
            prefetch_factor=(prefetch_factor if config['num_workers'] > 0 else None),
        )

    # Model
    print("🤖 Creating the original model...")
    model = build_model(config).to(device)

    # Optional checkpoint warm-start (kept from your version)
    ckpt_env = os.environ.get('LBP_LOAD_CKPT', '').strip()
    ckpt_default = os.path.join('./outputs_CKplus_cropped_adaptiveP', 'best_model.pth')
    if ckpt_env == '' or ckpt_env.upper() in ('NONE', 'NO'):
        ckpt_path = ''
    else:
        if ckpt_env and os.path.exists(ckpt_env):
            ckpt_path = ckpt_env
        else:
            ckpt_path = ckpt_default if os.path.exists(ckpt_default) else ''
    if ckpt_path:
        try:
            prev_train = model.training
            model.eval()
            H, W = get_image_hw(config)
            with torch.no_grad():
                _ = model(torch.zeros(8, 1, H, W, device=device))
            ckpt = torch.load(ckpt_path, map_location=device)
            raw_state = ckpt.get('model_state_dict', ckpt)
            incompatible = model.load_state_dict(raw_state, strict=False)
            print(f"📥 Loaded checkpoint (filtered rp_weights): {ckpt_path}\n   missing={len(incompatible.missing_keys)}, unexpected={len(incompatible.unexpected_keys)}")
            if prev_train: model.train()
            else: model.eval()
        except Exception as e:
            print(f"⚠️ Failed to load weights: {ckpt_path}, err={e}")

    if hasattr(model, 'fc_layers'):
        for m in model.fc_layers.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    # Info + paper accounting
    model_info = model.get_model_info()
    print("📋 Model info:")
    print(f"   Total parameters: {model_info['total_parameters']:,}")
    print(f"   LBP config: {model_info['lbp_config']}")
    print(f"   Block config: {model_info['block_config']}")
    H, W = get_image_hw(config)
    paper_size = get_paper_model_size_bytes(model)
    paper_ops = estimate_ops_paper(model, (1,1,H, W))
    print(f"[PAPER] size_bytes={paper_size} ({paper_size/1024:.2f} KB), gops={paper_ops['gops_total']:.6f}, ops(cmp/add/mul)={paper_ops['cmps']}/{paper_ops['adds']}/{paper_ops['muls']}")

    # -------------------- Adaptive-P stats (single-shot, no per-batch hooks) --------------------
    # We compute the P-map once from the global heatmap (static per-image), and derive per-epoch
    # counts by multiplying per-image counts by the number of images seen in that epoch.
    adaptive_p_all_epochs = []   # legacy container (unused now)
    static_per_image_p_counts = None
    try:
        first_lbp = None
        if hasattr(model, 'stages') and len(model.stages) > 0 and hasattr(model.stages[0], 'lbp_layer'):
            first_lbp = model.stages[0].lbp_layer
        if first_lbp is not None:
            # Only attempt JSON when adaptive-P is enabled on the first LBP layer
            ap_enabled = bool(getattr(first_lbp, 'adaptive_p_enable', False))
            per_image_json_dir = "./outputs_CKplus_cropped_adaptiveP"
            per_image_json_path = os.path.join(per_image_json_dir, 'adaptive_p_per_image.json')
            if not ap_enabled:
                # Avoid confusion with stale files from prior runs
                if os.path.exists(per_image_json_path):
                    print("ℹ️ Adaptive-P is disabled; not generating per-image adaptive-P JSON (existing file is from a previous run).")
                else:
                    print("ℹ️ Adaptive-P is disabled; skipping per-image adaptive-P JSON generation.")
            else:
                H, W = get_image_hw(config)
                # Compute mask on CPU to avoid GPU sync; same as used inside forward
                mask = first_lbp._get_adaptive_mask(H=H, W=W, device=torch.device('cpu'))
                if mask is not None and mask.numel() > 0:
                    # Derive per-pixel P(h,w) map with correct dimensions [H,W]
                    # mask has shape [P, N, H, W]; summing over N for any single pattern yields P(h,w)
                    try:
                        if hasattr(first_lbp, 'get_last_p_map'):
                            last_p = first_lbp.get_last_p_map()
                        else:
                            last_p = None
                    except Exception:
                        last_p = None

                    if last_p is not None and last_p.dim() == 3:
                        # last_p: [1, H, W] or [B, H, W] (when called after a forward)
                        if last_p.size(0) == 1:
                            p_map = last_p.squeeze(0).to(torch.int64)  # [H,W]
                        else:
                            # If batch-expanded, take the first image for a representative per-image save
                            p_map = last_p[0].to(torch.int64)
                    else:
                        # Fall back to computing from mask [P,N,H,W]
                        # Use the first pattern and sum across N to recover P(h,w)
                        if mask.dim() == 4:
                            p_map = mask[0].sum(dim=0).to(torch.int64)  # [H,W]
                        elif mask.dim() == 3:
                            # Rare case: [N,H,W]
                            p_map = mask.sum(dim=0).to(torch.int64)      # [H,W]
                        else:
                            raise RuntimeError(f"Unexpected adaptive mask shape: {tuple(mask.shape)}")
                    vals = p_map.detach().cpu().numpy().reshape(-1)
                    uniq, cnt = np.unique(vals, return_counts=True)
                    counts_by_value = {int(u): int(c) for u, c in zip(uniq, cnt)}
                    # Provide both a generic mapping and the legacy keys expected downstream (p2,p4,p6,p8,count)
                    total_px = int(H * W)
                    # Build dynamic per-P keys (prefixed 'p') for all observed P values
                    ap_vals_cfg = (config.get('lbp_layer', {}).get('adaptive_p_values')
                                   or config.get('adaptive_p', {}).get('values'))
                    dynamic_p_keys = {}
                    if ap_vals_cfg:
                        for v in ap_vals_cfg:
                            dynamic_p_keys[f'p{int(v)}'] = counts_by_value.get(int(v), 0)
                    else:
                        for v in counts_by_value.keys():
                            dynamic_p_keys[f'p{int(v)}'] = counts_by_value.get(int(v), 0)
                    static_per_image_p_counts = {
                        'by_value': counts_by_value,
                        'total': total_px,
                        'count': total_px,
                        **dynamic_p_keys
                    }
                    print(f"🧮 Adaptive-P per-image counts (static): {static_per_image_p_counts}")
                    # Immediately persist per-image artifacts (before training starts)
                    if SAVE_ENABLED:
                        try:
                            os.makedirs(per_image_json_dir, exist_ok=True)
                            # Counts JSON
                            with open(per_image_json_path, 'w') as jf:
                                json.dump({'per_image_counts': static_per_image_p_counts}, jf, indent=2)
                            # Config JSON
                            ap_cfg = {
                                'thresholds': config.get('lbp_layer', {}).get('adaptive_p_thresholds') or config.get('adaptive_p', {}).get('thresholds'),
                                'values': config.get('lbp_layer', {}).get('adaptive_p_values') or config.get('adaptive_p', {}).get('values'),
                                'seed': config.get('lbp_layer', {}).get('adaptive_perm_seed', 42),
                                'window': config.get('lbp_layer', {}).get('window', 5)
                            }
                            with open(os.path.join(per_image_json_dir, 'adaptive_p_config.json'), 'w') as jf2:
                                json.dump(ap_cfg, jf2, indent=2)
                            # Save P map and a visualization
                            np.save(os.path.join(per_image_json_dir, 'adaptive_p_map.npy'), p_map.detach().cpu().numpy())
                            try:
                                plt.figure(figsize=(4,3))
                                # Prefer configured adaptive_p_values for consistent scale bar ticks
                                ap_vals_cfg = (config.get('lbp_layer', {}).get('adaptive_p_values')
                                               or config.get('adaptive_p', {}).get('values'))
                                if ap_vals_cfg:
                                    ticks = sorted(int(v) for v in ap_vals_cfg)
                                else:
                                    ticks = sorted(int(k) for k in counts_by_value.keys()) if counts_by_value else [0,1]
                                vmin = min(ticks) if ticks else 0
                                vmax = max(ticks) if ticks else 1
                                plt.imshow(p_map.detach().cpu().numpy(), vmin=vmin, vmax=vmax, cmap='viridis')
                                cbar = plt.colorbar(label='P')
                                # Force discrete tick marks at the allowed P values (e.g., 4,8,12,16,20,24)
                                cbar.set_ticks(ticks)
                                cbar.set_ticklabels([str(t) for t in ticks])
                                plt.title('Adaptive P map (per-image)')
                                plt.tight_layout()
                                plt.savefig(os.path.join(per_image_json_dir, 'adaptive_p_map.png'))
                                plt.close()
                            except Exception as _e_pm:
                                print(f"⚠️ Failed to save adaptive_p_map.png: {_e_pm}")
                            # Save permutation indices per pattern (if available)
                            try:
                                perm = first_lbp.get_adaptive_permutation()
                                if perm is not None:
                                    np.save(os.path.join(per_image_json_dir, 'adaptive_permutation.npy'), perm.detach().cpu().numpy())
                            except Exception as _e_perm:
                                print(f"⚠️ Failed to save adaptive_permutation.npy: {_e_perm}")
                            print(f"📝 Saved per-image adaptive-P artifacts under: {per_image_json_dir}")
                        except Exception as je:
                            print(f"⚠️ Failed to write per-image adaptive-P artifacts: {je}")
                    else:
                        print("📝 Skipping per-image adaptive-P artifact save (saving disabled in debug mode)")
                else:
                    print("ℹ️ Adaptive-P mask is None (disabled or N!=8); skipping stats.")
        else:
            print("ℹ️ No stage-0 LBP layer found; skipping adaptive-P stats.")
    except Exception as e:
        print(f"⚠️ Failed to compute static adaptive-P per-image counts: {e}")

    # Optimizer param groups
    print("⚡ Creating optimizer...")
    base_lr = config.get('optim', {}).get('lr', config['training']['lr'])
    wd = config.get('optim', {}).get('weight_decay', config['training']['weight_decay'])
    lr_mult = config.get('optim', {}).get('lr_mult', {"gates": 2.0, "offsets": 2.0, "base": 1.0})

    gate_params = list(model.collect_gate_params()) if hasattr(model, 'collect_gate_params') else []
    offset_params = list(model.collect_offset_params()) if hasattr(model, 'collect_offset_params') else []
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

    # Scheduler / loss / AMP
    if config['training']['lr_scheduler'] == 'cosine':
        eta_min = float(config['training'].get('eta_min', config['training']['lr'] * 0.01))
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['training']['epochs'], eta_min=eta_min)
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    ls = float(config['training'].get('label_smoothing', 0.1))
    criterion = nn.CrossEntropyLoss(label_smoothing=ls)
    # GradScaler: first positional argument is init_scale; passing 'cuda' was incorrect (string becomes fill value)
    # Use default init_scale and enable only when on CUDA.
    scaler = GradScaler(enabled=True) if (config['amp'] and device.type == 'cuda') else None

    # Temp scheduler + EMA
    ts_cfg = config.get('temp_schedule', {})
    alpha_cfg = ts_cfg.get('alpha', {"start": 1.5, "end": 0.06, "mode": "cosine"})
    tau_cfg   = ts_cfg.get('tau',   {"start": 3.0, "end": 0.75, "mode": "cosine"})
    temp_sched = TemperatureScheduler(
        total_epochs=config['training']['epochs'],
        alpha_start=float(alpha_cfg.get('start', 1.5)), alpha_min=float(alpha_cfg.get('end', 0.06)),
        tau_start=float(tau_cfg.get('start', 3.0)),     tau_min=float(tau_cfg.get('end', 0.75)),
        mode=str(alpha_cfg.get('mode', 'cosine')),
        guard_freeze_alpha_epochs=0,
    )
    ema_cfg = config.get('ema', {"enable": True, "decay": 0.999})
    use_ema = bool(ema_cfg.get('enable', False))
    ema = ModelEMA(model, decay=float(ema_cfg.get('decay', 0.999))) if use_ema else None

    # Diagnostics helpers
    def set_bn_eval(m: nn.Module):
        for mod in m.modules():
            if isinstance(mod, nn.BatchNorm2d): mod.eval()
    def set_bn_train(m: nn.Module):
        for mod in m.modules():
            if isinstance(mod, nn.BatchNorm2d): mod.train()

    @torch.no_grad()
    def eval_accuracy(loader, soft_bits: bool, soft_gates: bool, hard_path: bool) -> float:
        if hard_path: model.eval()
        else:
            model.train(); set_bn_eval(model)
        if hasattr(model, 'set_ste'):
            model.set_ste(True, True)
        total, correct = 0, 0
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            # Older torch.cuda.amp.autocast does not support device_type kwarg; use enabled flag only
            with autocast(enabled=(config['amp'] and device.type=='cuda')):
                out = model(data)
            pred = out.argmax(1)
            total += target.size(0); correct += (pred == target).sum().item()
        return 100.0 * correct / max(1, total)

    @torch.no_grad()
    def bn_recalibrate_hard(loader, max_batches: int = 100):
        model.train(); set_bn_train(model)
        if hasattr(model, 'set_ste'):
            model.set_ste(True, True)
        seen = 0
        bn_old_moms = []
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                bn_old_moms.append((m, m.momentum)); m.momentum = 0.01
        for data, _ in loader:
            data = data.to(device); _ = model(data)
            seen += 1
            if seen >= max_batches: break
        for m, mom in bn_old_moms: m.momentum = mom
        model.eval()

    # Optional grid test / BN recal hooks preserved from your version...
    if os.environ.get('LBP_GRID_TEST', '0') == '1':
        print('🧪 2x2 soft/hard grid test (validation only):')
        model.eval()
        val_loader = DataLoader(
            get_mnist_datasets(config)[1],
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

    if os.environ.get('LBP_BN_RECAL', '0') == '1':
        print('🔧 Recalibrating BN statistics for hard path...')
        train_loader_small = DataLoader(
            get_mnist_datasets(config)[0],
            batch_size=config['training']['batch_size'], shuffle=True,
            num_workers=0, pin_memory=False
        )
        bn_recalibrate_hard(train_loader_small, max_batches=100)
        print('✅ BN recalibration complete')
        return model, None

    # Set STE switches + gate ste-scale
    if hasattr(model, 'set_ste'):
        model.set_ste(True, True)
    ste_scales = config.get('ste', {})
    if ste_scales:
        if hasattr(model, 'update_alpha') and ('lbp_ste_scale' in ste_scales):
            alpha_equiv = 1.0 / max(float(ste_scales.get('lbp_ste_scale', 6.0)), 1e-6)
            model.update_alpha(alpha_equiv)
        for s in model.stages:
            if hasattr(s, 'fuse') and hasattr(s.fuse, 'set_gate_ste_scale'):
                s.fuse.set_gate_ste_scale(float(ste_scales.get('gate_ste_scale', 6.0)))

    # Histories
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    alpha_hist, tau_hist, alive_hard_hist, guards_hist, epoch_times_hist = [], [], [], [], []
    epoch_status_lines = []

    best_val_acc, patience_counter = 0.0, 0
    output_dir = "./outputs_CKplus_cropped_adaptiveP"
    if SAVE_ENABLED:
        os.makedirs(output_dir, exist_ok=True)
    # Create a per-run id to disambiguate checkpoints from previous runs
    run_id = os.environ.get('RUN_ID', '').strip() or time.strftime('%Y%m%d_%H%M%S')
    print(f"🆔 Run ID: {run_id}")
    # Guard against stale best_model from a previous run
    best_model_path = os.path.join(output_dir, 'best_model.pth')
    if os.path.exists(best_model_path):
        if os.environ.get('RESET_BEST', '0') in ('1', 'true', 'True'):
            try:
                bak = os.path.join(output_dir, f"best_model.stale_{time.strftime('%Y%m%d_%H%M%S')}.pth")
                os.replace(best_model_path, bak)
                print(f"🧹 Found stale best_model.pth; moved to: {bak}")
            except Exception as _e:
                print(f"⚠️ Could not move stale best_model.pth: {_e}")
        else:
            print("⚠️ best_model.pth already exists. It may be from a previous run.\n"
                  "   To avoid mixing runs, either set RUN_ID and use a unique directory,\n"
                  "   or set RESET_BEST=1 to move the stale checkpoint aside before training.")
    base_lrs = None

    # ---- (One-time) window-mean distribution plot — only when adaptive-P is enabled ----
    try:
        ap_enabled = False
        try:
            _flbp = model.stages[0].lbp_layer if (hasattr(model, 'stages') and len(model.stages) > 0 and hasattr(model.stages[0], 'lbp_layer')) else None
            ap_enabled = bool(getattr(_flbp, 'adaptive_p_enable', False))
        except Exception:
            ap_enabled = False
        if window_mean_from_heatmap is not None and ap_enabled:
            H, W = get_image_hw(config)
            win_size = int(config.get('lbp_layer', {}).get('window', 5))
            # Prefer explicit lbp_layer heatmap_path; fall back to env
            hm_path = config.get('lbp_layer', {}).get('heatmap_path') or os.environ.get('GLOBAL_HEATMAP_PATH', None)
            # Compute local window mean (same setting as adaptive-P: exclude center)
            win_mean = window_mean_from_heatmap(
                path=hm_path,
                normalize="auto",
                device='cpu',
                target_hw=(H, W),
                kernel_size=win_size,
                exclude_center=True
            )  # [1,1,H,W]
            vals = win_mean.squeeze().detach().cpu().numpy().reshape(-1)

            # Histogram
            hist_path = os.path.join(output_dir, 'window_mean_hist.png')
            plt.figure(figsize=(6,4))
            plt.hist(vals, bins=100, range=(0.0, 1.0), density=True, alpha=0.8, color='#3b82f6')
            plt.xlabel('Local window mean')
            plt.ylabel('Density')
            plt.title(f'Window-mean distribution (kernel={win_size}x{win_size}, HxW={H}x{W})')
            # Draw configured thresholds if present
            thr = (config.get('lbp_layer', {}).get('adaptive_p_thresholds')
                   or config.get('adaptive_p', {}).get('thresholds'))
            if thr:
                for t in thr:
                    plt.axvline(float(t), color='red', linestyle='--', linewidth=1)
            plt.grid(True, alpha=0.25)
            plt.tight_layout()
            if SAVE_ENABLED:
                plt.savefig(hist_path)
                print(f"📝 Saved window-mean histogram: {hist_path}")
            else:
                print("📝 Skipping window-mean histogram save (saving disabled in debug mode)")
            plt.close()

            # CDF
            cdf_path = os.path.join(output_dir, 'window_mean_cdf.png')
            v_sorted = np.sort(vals)
            y = np.linspace(0.0, 1.0, num=v_sorted.size, endpoint=False)
            plt.figure(figsize=(6,4))
            plt.plot(v_sorted, y, label='Empirical CDF', color='#10b981')
            if thr:
                for t in thr:
                    plt.axvline(float(t), color='red', linestyle='--', linewidth=1)
            plt.xlabel('Local window mean')
            plt.ylabel('CDF')
            plt.title('Window-mean empirical CDF')
            plt.grid(True, alpha=0.25)
            plt.tight_layout()
            if SAVE_ENABLED:
                plt.savefig(cdf_path)
                print(f"📝 Saved window-mean CDF: {cdf_path}")
            else:
                print("📝 Skipping window-mean CDF save (saving disabled in debug mode)")
            plt.close()

            # Stats + bin proportions
            stats_path = os.path.join(output_dir, 'window_mean_stats.txt')
            if SAVE_ENABLED:
                with open(stats_path, 'w') as fstats:
                    qs = [1,5,10,25,50,75,90,95,99]
                    qv = np.percentile(vals, qs)
                    fstats.write('Quantiles (percentile: value)\n')
                    for p, v in zip(qs, qv):
                        fstats.write(f"  {p:>2}% : {v:.6f}\n")
                    # Generalize bin proportions for any number of thresholds
                    if thr and isinstance(thr, (list, tuple)) and len(thr) >= 1:
                        thr_sorted = sorted([float(t) for t in thr])
                        # Build segments: (-inf, t0), [t0, t1), ..., [t_{k-1}, +inf)
                        seg_masks = []
                        prev = -np.inf
                        for t in thr_sorted:
                            seg_masks.append((vals >= prev) & (vals < t))
                            prev = t
                        seg_masks.append(vals >= prev)
                        props = [float(np.mean(m)) for m in seg_masks]
                        fstats.write('\nBin proportions given thresholds ' + str(thr_sorted) + '\n')
                        # If adaptive_p_values present, map segments to P values; else label generically
                        ap_vals = (config.get('lbp_layer', {}).get('adaptive_p_values')
                                   or config.get('adaptive_p', {}).get('values'))
                        if ap_vals and len(ap_vals) == len(props):
                            for P, pr in zip(ap_vals, props):
                                fstats.write(f"  P={int(P)}: {pr:.6f}\n")
                        else:
                            for i, pr in enumerate(props):
                                fstats.write(f"  bin[{i}]: {pr:.6f}\n")
            else:
                print("📝 Skipping window-mean stats file save (saving disabled in debug mode)")
        else:
            if not ap_enabled:
                print("ℹ️ Adaptive-P is disabled; skipping window-mean plots")
            else:
                print("ℹ️ window_mean_from_heatmap unavailable; skipping window-mean plots")
    except Exception as e:
        print(f"⚠️ Failed to compute/save window-mean plots: {e}")

    print("🎯 Starting training...")
    # CUDA sync helper for accurate timings
    def _sync_cuda():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
    for epoch in range(config['training']['epochs']):
        epoch_start_time = time.time()

        # (no per-batch collection)

        # alpha/tau schedule → alpha and gate_ste_scale
        alpha, tau = temp_sched.step(epoch)
        if hasattr(model, 'update_alpha'):
            model.update_alpha(float(alpha))
        gate_ste_scale = 1.0 / max(float(tau), 1e-6)
        for s in getattr(model, 'stages', []):
            if hasattr(s, 'fuse') and hasattr(s.fuse, 'set_gate_ste_scale'):
                s.fuse.set_gate_ste_scale(gate_ste_scale)

        # freeze windows
        freeze_offsets_epochs = config['freeze_schedule']['freeze_offsets_epochs']
        freeze_gates_extra = config['freeze_schedule']['freeze_gates_extra_epochs']
        for name, p in model.named_parameters():
            if 'offsets_raw' in name:
                p.requires_grad = epoch >= freeze_offsets_epochs
            if name.endswith('.alpha') or ('alpha' in name):
                p.requires_grad = epoch >= freeze_offsets_epochs
            if 'gate_logits' in name:
                p.requires_grad = epoch >= (freeze_offsets_epochs + freeze_gates_extra)

        # early hard-finetune phase
        hf = config.get('hard_finetune', {"enable": False})
        hard_phase = hf.get('enable', False) and (epoch >= hf.get('start_epoch', 10**9))
        if hard_phase:
            for pg in optimizer.param_groups:
                pg['lr'] = pg['lr'] * float(hf.get('lr_mult', 0.1))
            if hasattr(model, 'set_ste'):
                model.set_ste(use_ste_bits=hf.get('hard_forward_bits', True),
                              use_ste_gates=hf.get('hard_forward_gates', True))
            set_bn_mode(model, hf.get('bn_mode', 'track'))
        else:
            if hasattr(model, 'set_ste'):
                model.set_ste(use_ste_bits=True, use_ste_gates=True)

        # train epoch
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        # warmup
        if base_lrs is None:
            base_lrs = [g['lr'] for g in optimizer.param_groups]
        warmup_epochs = config['training'].get('warmup_epochs', 0)
        if epoch < warmup_epochs:
            warmup_factor = (epoch + 1) / max(1, warmup_epochs)
            for pg, base_lr in zip(optimizer.param_groups, base_lrs):
                pg['lr'] = base_lr * warmup_factor

        # Timing accumulators per epoch
        data_time_sum = 0.0
        fwd_time_sum = 0.0
        bwd_opt_time_sum = 0.0

        # Epoch-level debug accumulators (for optional reporting)
        offset_radius_stats = None  # will hold mean/std/edge/center stats for stage0 offsets
        adaptive_p_hist = None      # histogram of adaptive P usage (stage0)

        for batch_idx, (data, target) in enumerate(train_loader):
            t0 = time.time()
            # Faster H2D transfers when pin_memory=True
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            _sync_cuda(); t1 = time.time()
            data_time_sum += (t1 - t0)
            optimizer.zero_grad()
            if config['amp'] and scaler is not None:
                with autocast(enabled=(config['amp'] and device.type=='cuda')):
                    tf0 = time.time()
                    output = model(data)
                    loss = criterion(output, target) + model.get_offset_penalty()
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
                    loss = loss + 1e-3 * dist_reg
                _sync_cuda(); tf1 = time.time(); fwd_time_sum += (tf1 - tf0)
                tb0 = time.time()
                scaler.scale(loss).backward()
                if config['gradient_clipping']['enabled']:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clipping']['max_norm'])
                scaler.step(optimizer); scaler.update()
                if ema is not None: ema.update(model)
                _sync_cuda(); tb1 = time.time(); bwd_opt_time_sum += (tb1 - tb0)
            else:
                tf0 = time.time()
                output = model(data)
                loss = criterion(output, target) + model.get_offset_penalty()
                if config.get('gate_sparsity', {}).get('enable', False):
                    target_alive = config['gate_sparsity'].get('target_alive', 0.5)
                    lambda_g = config['gate_sparsity'].get('weight', 1e-3)
                    gate_means = []
                    for s in model.stages:
                        if hasattr(s, 'get_gate_values'):
                            g = s.get_gate_values()
                            if g is not None: gate_means.append(g.mean())
                    if gate_means:
                        gate_mean = torch.stack(gate_means).mean()
                        loss = loss + lambda_g * F.l1_loss(gate_mean, torch.tensor(target_alive, device=gate_mean.device))
                with torch.no_grad():
                    p_batch = F.softmax(output, dim=1).mean(0)
                uniform = torch.full_like(p_batch, 1.0 / config['head']['num_classes'])
                dist_reg = F.kl_div((p_batch + 1e-8).log(), uniform, reduction='batchmean')
                loss = loss + 1e-3 * dist_reg
                _sync_cuda(); tf1 = time.time(); fwd_time_sum += (tf1 - tf0)
                tb0 = time.time()
                loss.backward()
                if config['gradient_clipping']['enabled']:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clipping']['max_norm'])
                optimizer.step()
                if ema is not None: ema.update(model)
                _sync_cuda(); tb1 = time.time(); bwd_opt_time_sum += (tb1 - tb0)

            train_loss += float(loss.item())
            _, predicted = output.max(1)
            train_total += target.size(0)
            train_correct += int(predicted.eq(target).sum().item())

        train_loss /= len(train_loader)

        # === Epoch-end debug: offset drift & adaptive-P usage ===
        try:
            # Offset radius monitoring (stage 0 only)
            first_lbp = None
            if hasattr(model, 'stages') and len(model.stages) > 0 and hasattr(model.stages[0], 'lbp_layer'):
                first_lbp = model.stages[0].lbp_layer
            if first_lbp is not None:
                off = first_lbp.get_offsets()  # [P, N, 2]
                r = off.norm(dim=-1)           # [P, N]
                radius = float(first_lbp.radius) if hasattr(first_lbp, 'radius') else 1.0
                mean_r = float(r.mean().item())
                std_r = float(r.std().item())
                frac_edge = float((r > 0.9 * radius).float().mean().item())
                frac_center = float((r < 0.1 * radius).float().mean().item())
                offset_radius_stats = {
                    'mean': mean_r,
                    'std': std_r,
                    'frac_edge': frac_edge,
                    'frac_center': frac_center,
                    'radius': radius
                }
                print(f"   🔍 Offsets(radius): mean={mean_r:.4f} std={std_r:.4f} edge={frac_edge:.2%} center={frac_center:.2%}")
            # Adaptive-P histogram (from last P map)
            if first_lbp is not None and getattr(first_lbp, 'adaptive_p_enable', False):
                p_map = first_lbp.get_last_p_map()
                if p_map is not None:
                    # p_map shape: [B,H,W]
                    vals = p_map.detach().cpu().numpy().reshape(-1)
                    unique, counts = np.unique(vals, return_counts=True)
                    adaptive_p_hist = {int(u): int(c) for u, c in zip(unique, counts)}
                    # Compact textual summary
                    parts = ', '.join([f"P={u}:{c}" for u, c in adaptive_p_hist.items()])
                    print(f"   🔍 Adaptive-P usage (last batch): {parts}")
        except Exception as _e_dbg:
            print(f"   ⚠️ Debug stats (offset/adaptive-P) failed: {_e_dbg}")
        train_acc = 100. * train_correct / max(1, train_total)

        # validation (optionally less frequent via VALIDATE_EVERY)
        model.eval()
        do_validate = (val_loader is not None) and (((epoch + 1) % validate_every == 0) or ((epoch + 1) == config['training']['epochs']))
        if do_validate:
            val_loss, val_correct, val_total = 0.0, 0, 0
            pred_hist = torch.zeros(config['head']['num_classes'], dtype=torch.long)
            with torch.no_grad():
                ema_backup = None
                if use_ema and (ema is not None):
                    ema_backup = ema.apply_to(model)
                for data, target in val_loader:
                    data = data.to(device, non_blocking=True)
                    target = target.to(device, non_blocking=True)
                    with autocast(enabled=(config['amp'] and device.type=='cuda')):
                        output = model(data)
                        loss = criterion(output, target) + model.get_offset_penalty()
                    val_loss += float(loss.item())
                    _, predicted = output.max(1)
                    val_total += target.size(0)
                    val_correct += int(predicted.eq(target).sum().item())
                    pred_hist += torch.bincount(predicted.detach().cpu(), minlength=config['head']['num_classes'])
                if ema_backup is not None:
                    ModelEMA.restore(model, ema_backup)
            if hard_phase:
                bn_recalibrate_hard(train_loader, max_batches=int(config.get('hard_finetune', {}).get('bn_recal_batches', 100)))
            val_loss /= max(1, len(val_loader))
            val_acc = 100. * val_correct / max(1, val_total)

            # Optional: evaluate a "test-like" validation metric (hard path + BN recalibration)
            # Enable with VAL_HARD_BN=1 and optionally VAL_BN_BATCHES (default 100)
            val_hard_bn_acc = None
            try:
                if os.environ.get('VAL_HARD_BN', '0') in ('1', 'true', 'True'):
                    with torch.no_grad():
                        # Optionally use EMA weights as in the standard val (keep consistent)
                        ema_backup2 = None
                        if use_ema and (ema is not None):
                            ema_backup2 = ema.apply_to(model)

                        # Hard-path BN recalibration using train data (small subset)
                        recal_batches = int(os.environ.get('VAL_BN_BATCHES', '100') or '100')
                        bn_recalibrate_hard(train_loader, max_batches=recal_batches)

                        # Evaluate on val in eval() mode
                        vh_correct, vh_total = 0, 0
                        for data, target in val_loader:
                            data = data.to(device, non_blocking=True)
                            target = target.to(device, non_blocking=True)
                            with autocast(enabled=(config['amp'] and device.type=='cuda')):
                                out = model(data)
                            pred = out.argmax(1)
                            vh_total += target.size(0)
                            vh_correct += int((pred == target).sum().item())
                        val_hard_bn_acc = 100.0 * vh_correct / max(1, vh_total)

                        # Restore model if EMA was applied
                        if ema_backup2 is not None:
                            ModelEMA.restore(model, ema_backup2)
            except Exception as _e:
                print(f"   ⚠️ VAL_HARD_BN evaluation failed: {_e}")
        else:
            val_loss, val_acc = train_loss, train_acc
            pred_hist = torch.zeros(config['head']['num_classes'], dtype=torch.long)

        scheduler.step()

        # collapse guard
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
                    best_model_path = os.path.join(output_dir, 'best_model.pth')
                    if os.path.exists(best_model_path):
                        try:
                            ckpt = torch.load(best_model_path, map_location=device)
                            model.load_state_dict(ckpt.get('model_state_dict', ckpt), strict=False)
                            if ('ema_shadow' in ckpt) and (ema is not None) and ckpt['ema_shadow']:
                                ema.shadow = ckpt['ema_shadow']
                            print("   🛡️ Collapse guard: restored from best and softened tau/froze alpha briefly")
                        except Exception as e:
                            print(f"   ⚠️ Collapse guard restore failed: {e}")

        # record
        train_losses.append(train_loss); val_losses.append(val_loss)
        train_accs.append(train_acc);   val_accs.append(val_acc)

        epoch_time = time.time() - epoch_start_time

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
            try:
                if 'val_hard_bn_acc' in locals() and (val_hard_bn_acc is not None):
                    print(f"   Val (hard+BN recal) accuracy: {val_hard_bn_acc:.2f}%")
            except Exception:
                pass
        else:
            alive_hard = mean_alive
            print(f"\n📊 Epoch {epoch+1}/{config['training']['epochs']} summary:")
            print(f"   Train loss: {train_loss:.4f}, Train accuracy: {train_acc:.2f}%")
            print(f"   Val loss: {val_loss:.4f}, Val accuracy: {val_acc:.2f}%")
            print(f"   Current Alpha: {alpha:.4f}, Tau: {tau:.4f}, Alive(hard): {alive_hard:.3f}, guards={guard_events}")
            try:
                if 'val_hard_bn_acc' in locals() and (val_hard_bn_acc is not None):
                    print(f"   Val (hard+BN recal) accuracy: {val_hard_bn_acc:.2f}%")
            except Exception:
                pass
        print(f"   Validation prediction histogram: {[int(x) for x in pred_hist.tolist()]}")
        # Timing breakdown summary (data vs. compute)
        n_batches = max(1, len(train_loader))
        print(f"   Time elapsed: {epoch_time:.2f}s | avg/batch: {epoch_time / n_batches:.3f}s")
        if (data_time_sum + fwd_time_sum + bwd_opt_time_sum) > 0:
            other = max(0.0, epoch_time - (data_time_sum + fwd_time_sum + bwd_opt_time_sum))
            total_t = max(1e-6, data_time_sum + fwd_time_sum + bwd_opt_time_sum + other)
            print(
                f"   ⏱️ breakdown: data={data_time_sum:.2f}s ({100*data_time_sum/total_t:.1f}%), "
                f"fwd={fwd_time_sum:.2f}s ({100*fwd_time_sum/total_t:.1f}%), "
                f"bwd+opt={bwd_opt_time_sum:.2f}s ({100*bwd_opt_time_sum/total_t:.1f}%), "
                f"other={other:.2f}s ({100*other/total_t:.1f}%)"
            )

        alpha_hist.append(float(alpha))
        tau_hist.append(float(tau))
        alive_hard_hist.append(float(alive_hard))
        guards_hist.append(int(guard_events))
        epoch_times_hist.append(float(epoch_time))
        epoch_status_lines.append(
            f"Epoch {epoch+1}: Current Alpha: {alpha:.4f}, Tau: {tau:.4f}, Alive(hard): {alive_hard:.3f}, guards={guard_events}, time={epoch_time:.2f}s"
        )

        if gstats:
            stab = config.get('stability', {})
            temp_sched.update_on_val(val_acc, alive_hard,
                                     acc_drop_thr=float(stab.get('acc_drop_thr', 0.05)),
                                     alive_upper=float(stab.get('alive_upper', 0.26)))

        # ---------------- record adaptive-P stats (derived from static per-image counts) ----------------
        try:
            if static_per_image_p_counts is not None:
                n_imgs = len(train_dataset)
                # Dynamically derive all P keys from by_value plus legacy keys
                by_val = static_per_image_p_counts.get('by_value', {})
                # Prefer configured adaptive_p_values ordering if available
                ap_vals_cfg = (config.get('lbp_layer', {}).get('adaptive_p_values')
                               or config.get('adaptive_p', {}).get('values'))
                if ap_vals_cfg:
                    ordered_P = [int(v) for v in ap_vals_cfg if int(v) in by_val]
                else:
                    ordered_P = sorted(int(k) for k in by_val.keys())
                epoch_counts = {f"p{int(P)}": int(by_val[int(P)]) * n_imgs for P in ordered_P}
                # Maintain 'count' total pixels * n_imgs
                epoch_counts["count"] = int(static_per_image_p_counts.get("count", static_per_image_p_counts.get('total', 0))) * n_imgs
                adaptive_p_all_epochs.append({
                    "epoch": int(epoch + 1),
                    "counts": epoch_counts
                })
                # Human-readable summary
                parts = ', '.join([f"P={int(P)}:{epoch_counts[f'p{int(P)}']}" for P in ordered_P])
                print(f"   📝 Filled adaptive-P stats for epoch {epoch+1}: {parts}")
        except Exception as e:
            print(f"   ⚠️ Failed to fill adaptive-P stats for epoch {epoch+1}: {e}")

        # Persist debug stats (optional) under SAVE_ENABLED
        if SAVE_ENABLED:
            try:
                dbg_dir = os.path.join(output_dir, 'debug_stats')
                os.makedirs(dbg_dir, exist_ok=True)
                # Offset radius log (append CSV)
                if offset_radius_stats is not None:
                    or_path = os.path.join(dbg_dir, 'offset_radius.csv')
                    write_header = (not os.path.exists(or_path))
                    with open(or_path, 'a') as f_or:
                        if write_header:
                            f_or.write('epoch,mean,std,frac_edge,frac_center,radius\n')
                        f_or.write(f"{epoch+1},{offset_radius_stats['mean']:.6f},{offset_radius_stats['std']:.6f},{offset_radius_stats['frac_edge']:.6f},{offset_radius_stats['frac_center']:.6f},{offset_radius_stats['radius']:.6f}\n")
                # Adaptive-P histogram log (append CSV)
                if adaptive_p_hist is not None:
                    ap_path = os.path.join(dbg_dir, 'adaptive_p_hist.csv')
                    write_header = (not os.path.exists(ap_path))
                    with open(ap_path, 'a') as f_ap:
                        if write_header:
                            f_ap.write('epoch,' + ','.join([f"P{p}" for p in sorted(adaptive_p_hist.keys())]) + '\n')
                        row = [str(epoch+1)] + [str(adaptive_p_hist.get(p, 0)) for p in sorted(adaptive_p_hist.keys())]
                        f_ap.write(','.join(row) + '\n')
            except Exception as _e_save_dbg:
                print(f"   ⚠️ Failed to persist debug stats: {_e_save_dbg}")

        # best save / patience
        if val_acc > best_val_acc + config['training']['min_delta']:
            best_val_acc = val_acc
            patience_counter = 0
            if SAVE_ENABLED:
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
                    'config': config,
                    'run_id': run_id
                }
                torch.save(ckpt_extra, best_model_path)
                print(f"   🎉 New best model saved: {best_model_path}")
            else:
                print("   🎉 New best model found (saving disabled in debug mode)")
        else:
            patience_counter += 1
            print(f"   ⏳ Patience count: {patience_counter}/{config['training']['patience']}")

        if patience_counter >= config['training']['patience']:
            print(f"\n🛑 Early stopping triggered! No improvement for {config['training']['patience']} epochs")
            # Use a flag to exit the outer epoch loop cleanly (avoid break outside loop if indentation drifted)
            early_stop = True
        else:
            early_stop = False

        print("-" * 80)

        if early_stop:
            break

    print(f"\n🎉 Training complete! Best validation accuracy: {best_val_acc:.2f}%")

    # final save (disabled in debug unless SAVE_ENABLED)
    if SAVE_ENABLED:
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
            'config': config,
            'run_id': run_id
        }, final_model_path)
        print(f"📁 Final model saved: {final_model_path}")
    else:
        print("📁 Skipping final model save (saving disabled in debug mode)")

    # logs/plots (disabled in debug unless SAVE_ENABLED)
    if SAVE_ENABLED:
        try:
            os.makedirs(output_dir, exist_ok=True)
            csv_path = os.path.join(output_dir, 'training_log.csv')
            txt_path = os.path.join(output_dir, 'training_log.txt')
            status_txt_path = os.path.join(output_dir, 'epoch_status_log.txt')
            loss_plot_path = os.path.join(output_dir, 'loss_plot.png')
            acc_plot_path = os.path.join(output_dir, 'acc_plot.png')

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

            with open(status_txt_path, 'w') as fs:
                fs.write("Per-epoch status lines (alpha/tau/alive_hard/guards/time):\n")
                for line in epoch_status_lines:
                    fs.write(line + "\n")

            # (No adaptive-P JSON at the end; per-image JSON was saved before training.)

            try:
                epochs = list(range(1, n+1))
                plt.figure()
                if len(train_losses) > 0:
                    plt.plot(epochs[:len(train_losses)], train_losses, label='train_loss')
                if len(val_losses) > 0:
                    plt.plot(epochs[:len(val_losses)], val_losses, label='val_loss')
                plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Training and Validation Loss')
                plt.legend(); plt.grid(True); plt.tight_layout(); plt.savefig(loss_plot_path); plt.close()

                plt.figure()
                if len(train_accs) > 0:
                    plt.plot(epochs[:len(train_accs)], train_accs, label='train_acc')
                if len(val_accs) > 0:
                    plt.plot(epochs[:len(val_accs)], val_accs, label='val_acc')
                plt.xlabel('Epoch'); plt.ylabel('Accuracy (%)'); plt.title('Training and Validation Accuracy')
                plt.legend(); plt.grid(True); plt.tight_layout(); plt.savefig(acc_plot_path); plt.close()
            except Exception as e:
                print(f"⚠️ Failed to save plots: {e}")

            print(f"✅ Training logs and plots saved in: {output_dir}")
            print(f"   - CSV: {csv_path}")
            print(f"   - TXT (table): {txt_path}")
            print(f"   - Status lines: {status_txt_path}")
        except Exception as e:
            print(f"⚠️ Failed to write training logs/plots to {output_dir}: {e}")
    else:
        print("📝 Skipping logs and plots save (saving disabled in debug mode)")

    # Final: load best → BN recal → test (optionally compare with no-recal)
    try:
        best_model_path = os.path.join(output_dir, 'best_model.pth')
        if os.path.exists(best_model_path):
            print("===> Final evaluation: load best weights, then TEST ...")
            ckpt = torch.load(best_model_path, map_location=device)
            ckpt_run_id = ckpt.get('run_id')
            prev_mode = model.training
            model.eval()
            with torch.no_grad():
                H, W = get_image_hw(config)
                _ = model(torch.zeros(8, 1, H, W, device=device))
            if (ckpt_run_id is None) or (ckpt_run_id != run_id):
                if os.environ.get('OVERRIDE_STALE_BEST', '0') in ('1', 'true', 'True'):
                    print(f"⚠️ Loading best_model.pth from a different run (ckpt_run_id={ckpt_run_id}). OVERRIDE_STALE_BEST=1 set.")
                    raw_state = ckpt.get('model_state_dict', ckpt)
                    _incompat = model.load_state_dict(raw_state, strict=False)
                else:
                    print(f"⚠️ best_model.pth appears to be from a previous run (ckpt_run_id={ckpt_run_id}, current={run_id}).\n"
                          "   Skipping stale checkpoint for final evaluation; using current in-memory weights instead.")
            else:
                raw_state = ckpt.get('model_state_dict', ckpt)
                _incompat = model.load_state_dict(raw_state, strict=False)
            if hasattr(model, 'set_ste'):
                model.set_ste(True, True)
            def _eval_on(loader):
                total_loss = 0.0; total_correct = 0; total = 0
                model.eval()
                with torch.no_grad():
                    for data, target in loader:
                        data, target = data.to(device), target.to(device)
                        with autocast(enabled=(config['amp'] and device.type=='cuda')):
                            out = model(data)
                            loss = criterion(out, target)
                        total_loss += float(loss.item())
                        pred = out.argmax(1)
                        total += int(target.size(0))
                        total_correct += int((pred == target).sum().item())
                avg_loss = total_loss / max(1, len(loader))
                acc = 100.0 * total_correct / max(1, total)
                return acc, avg_loss

            test_loader_final = DataLoader(test_dataset, batch_size=config['training']['batch_size'], shuffle=False, num_workers=0)

            mode = os.environ.get('TEST_EVAL_MODE', '').strip().lower()  # '', 'recal' (default), 'norecal', 'both'
            if mode in ('norecal', 'both'):
                acc_nr, loss_nr = _eval_on(test_loader_final)
                print(f"[FINAL] (no BN recal)  test_acc={acc_nr:.4f}%, test_loss={loss_nr:.4f}")

            # BN recalibrate unless explicitly disabled
            if mode != 'norecal':
                print("   → BN recalibration before final test ...")
                calib_loader = DataLoader(
                    train_dataset,
                    batch_size=config['training']['batch_size'],
                    shuffle=True, num_workers=0, pin_memory=False
                )
                bn_recalibrate_hard(calib_loader, max_batches=int(config.get('hard_finetune', {}).get('bn_recal_batches', 200)))
                acc_r, loss_r = _eval_on(test_loader_final)
                print(f"[FINAL] (with BN recal) test_acc={acc_r:.4f}%, test_loss={loss_r:.4f}")
            if prev_mode: model.train()
    except Exception as _e:
        print(f"[FINAL] Final evaluation failed: {_e}")

    # (no hook to clean up)

    return model, best_val_acc

if __name__ == "__main__":
    train_original_model()
