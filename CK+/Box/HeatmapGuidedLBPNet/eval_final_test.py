#!/usr/bin/env python3
"""
Evaluate final test accuracy on a trained LBPNet model.
Loads the best model checkpoint and reports test accuracy with BN recalibration.
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
import argparse

from lbpnet.models import build_model
from lbpnet.data import get_svhn_datasets


def get_image_hw(cfg):
    """Get image height and width from config."""
    sz = cfg.get('image_size', 32)
    if isinstance(sz, (list, tuple)) and len(sz) == 2:
        return int(sz[0]), int(sz[1])
    return int(sz), int(sz)


def set_bn_train(model):
    """Set all BatchNorm layers to training mode."""
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.train()


def bn_recalibrate_hard(model, loader, device, max_batches=200, use_amp=False):
    """Recalibrate BatchNorm statistics using training data."""
    print(f"   → BN recalibration ({max_batches} batches)...")
    
    model.train()
    set_bn_train(model)
    
    if hasattr(model, 'set_ste'):
        model.set_ste(True, True)
    
    # Save old momentum values
    bn_old_moms = []
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            bn_old_moms.append((m, m.momentum))
            m.momentum = 0.01
    
    # Run batches through model
    seen = 0
    with torch.no_grad():
        for data, _ in loader:
            data = data.to(device)
            with autocast(enabled=use_amp):
                _ = model(data)
            seen += 1
            if seen % 50 == 0:
                print(f"      Progress: {seen}/{max_batches} batches")
            if seen >= max_batches:
                break
    
    # Restore momentum
    for m, mom in bn_old_moms:
        m.momentum = mom
    
    model.eval()
    print(f"   ✓ BN recalibration complete")


def evaluate_model(model, loader, criterion, device, use_amp=False):
    """Evaluate model on given loader."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            
            with autocast(enabled=use_amp):
                out = model(data)
                loss = criterion(out, target)
            
            total_loss += float(loss.item())
            pred = out.argmax(1)
            total += int(target.size(0))
            total_correct += int((pred == target).sum().item())
    
    avg_loss = total_loss / max(1, len(loader))
    acc = 100.0 * total_correct / max(1, total)
    
    return acc, avg_loss


def main():
    parser = argparse.ArgumentParser(description='Evaluate final test accuracy')
    parser.add_argument('--output_dir', type=str, default='outputs_svhn_cropped_hm_P_decay_less_channel3_pat8',
                        help='Directory containing best_model.pth and config.yaml')
    parser.add_argument('--bn_recal_batches', type=int, default=200,
                        help='Number of batches for BN recalibration')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size for evaluation (default: use training batch size)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (e.g., cuda:0, cpu)')
    parser.add_argument('--mode', type=str, default='both', choices=['recal', 'norecal', 'both'],
                        help='Evaluation mode: recal (with BN), norecal (without BN), both')
    
    args = parser.parse_args()
    
    # Load checkpoint first to get config
    model_path = os.path.join(args.output_dir, 'best_model.pth')
    if not os.path.exists(model_path):
        print(f"❌ Model checkpoint not found: {model_path}")
        sys.exit(1)
    
    print(f"📂 Loading checkpoint: {model_path}")
    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # Try to get config from checkpoint, fallback to yaml, then preset
    if 'config' in ckpt:
        print(f"📄 Using config from checkpoint")
        config = ckpt['config']
    else:
        config_path = os.path.join(args.output_dir, 'config.yaml')
        if os.path.exists(config_path):
            print(f"📄 Loading config from: {config_path}")
            with open(config_path, 'r') as f:
                import yaml
                config = yaml.safe_load(f)
        else:
            print(f"⚠️  Config not found in checkpoint or yaml")
            print(f"📝 Using default preset config...")
            from train_original_model_cropped import PRESETS
            config = PRESETS.get('paper_svhn_rp_cropped_P')
            if not config:
                print("❌ Could not load default config")
                sys.exit(1)
    
    # Setup device first
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    print(f"🖥️  Using device: {device}")
    
    # Build model
    print("🔨 Building model...")
    model = build_model(config)
    model = model.to(device)
    
    # Load state dict
    if 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    else:
        state_dict = ckpt
    
    # Move state dict to device
    state_dict = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                  for k, v in state_dict.items()}
    
    # Handle rp_map_idx mismatch by filtering them out and loading manually
    # The model initializes these as empty buffers, but checkpoint has the actual indices
    rp_map_keys = [k for k in state_dict.keys() if 'rp_map_idx' in k]
    if rp_map_keys:
        print(f"⚠️  Found {len(rp_map_keys)} RP map indices in checkpoint")
        # Extract and temporarily remove rp_map_idx from state_dict
        rp_maps = {k: state_dict.pop(k) for k in rp_map_keys}
    
    # Load the rest of the state dict
    incompatible = model.load_state_dict(state_dict, strict=False)
    
    # Manually set the rp_map_idx buffers
    if rp_map_keys:
        for name, param in rp_maps.items():
            # Navigate to the buffer and resize+copy
            parts = name.split('.')
            module = model
            for part in parts[:-1]:
                module = getattr(module, part)
            # The buffer name is the last part
            buffer_name = parts[-1]
            if hasattr(module, buffer_name):
                # Resize the buffer and copy values
                getattr(module, buffer_name).resize_(param.shape)
                getattr(module, buffer_name).copy_(param)
        print(f"✓ Manually loaded {len(rp_maps)} RP map indices")
    
    if incompatible.missing_keys:
        print(f"⚠️  Missing keys: {incompatible.missing_keys[:5]}...")
    if incompatible.unexpected_keys:
        print(f"⚠️  Unexpected keys: {incompatible.unexpected_keys[:5]}...")
    
    print("✓ Model weights loaded successfully")
    
    # Set STE mode if available
    if hasattr(model, 'set_ste'):
        model.set_ste(True, True)
    
    # Warm up model
    model.eval()
    H, W = get_image_hw(config)
    with torch.no_grad():
        _ = model(torch.zeros(8, 1, H, W, device=device))
    
    # Load datasets
    print("📊 Loading datasets...")
    train_dataset, val_dataset, test_dataset = get_svhn_datasets(config)
    
    # Determine batch size
    batch_size = args.batch_size if args.batch_size else config['training']['batch_size']
    
    # Create dataloaders
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=0, pin_memory=False)
    
    # Setup loss (with label smoothing if specified in config)
    label_smoothing = config.get('training', {}).get('label_smoothing', 0.0)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    if label_smoothing > 0:
        print(f"📊 Using label smoothing: {label_smoothing}")
    
    # Check AMP setting
    use_amp = config.get('amp', False) and device.type == 'cuda'
    
    print("\n" + "="*60)
    print("FINAL TEST EVALUATION")
    print("="*60)
    
    # Evaluate without BN recalibration
    if args.mode in ('norecal', 'both'):
        print("\n📈 Evaluating WITHOUT BN recalibration...")
        acc_nr, loss_nr = evaluate_model(model, test_loader, criterion, device, use_amp)
        print(f"[FINAL] (no BN recal)  test_acc={acc_nr:.4f}%, test_loss={loss_nr:.4f}")
    
    # Evaluate with BN recalibration
    if args.mode in ('recal', 'both'):
        print("\n📈 Evaluating WITH BN recalibration...")
        bn_recalibrate_hard(model, train_loader, device, 
                           max_batches=args.bn_recal_batches, use_amp=use_amp)
        acc_r, loss_r = evaluate_model(model, test_loader, criterion, device, use_amp)
        print(f"[FINAL] (with BN recal) test_acc={acc_r:.4f}%, test_loss={loss_r:.4f}")
    
    print("\n" + "="*60)
    print("✅ Evaluation complete!")
    print("="*60)


if __name__ == "__main__":
    main()
