#!/usr/bin/env python3
"""
Check architecture compatibility with cropped SVHN images (29x18)
This script diagnoses potential issues with downsampling and spatial dimensions.
"""

import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lbpnet.models import build_model
from train_original_model import PRESETS

def check_spatial_dimensions(config, image_size=(29, 18)):
    """
    Trace spatial dimensions through the network to detect problems.
    """
    print(f"\n{'='*70}")
    print(f"ARCHITECTURE ANALYSIS FOR {image_size} INPUT")
    print(f"{'='*70}")
    
    # Get downsample stages
    downsample_at = set(config['blocks'].get('downsample_at', []))
    num_stages = config['blocks']['stages']
    
    h, w = image_size
    print(f"\nInput size: H={h}, W={w}")
    print(f"Number of stages: {num_stages}")
    print(f"Downsample at stages: {sorted(downsample_at)}")
    
    print(f"\n{'Stage':<8} {'Downsample?':<12} {'H':<6} {'W':<6} {'Issue?'}")
    print(f"{'-'*50}")
    
    issues = []
    
    for stage in range(num_stages):
        downsample = stage in downsample_at
        
        # Before this stage
        before_h, before_w = h, w
        
        # After downsample (if applicable)
        if downsample:
            h = h // 2
            w = w // 2
        
        issue = ""
        if h < 1 or w < 1:
            issue = "⚠️ DIMENSION TOO SMALL!"
            issues.append(f"Stage {stage}: Dimension collapsed to {h}x{w}")
        elif h < 3 or w < 3:
            issue = "⚠️ TOO SMALL FOR LBP!"
            issues.append(f"Stage {stage}: Dimension {h}x{w} may be too small for effective LBP")
        
        print(f"{stage:<8} {'Yes' if downsample else 'No':<12} {h:<6} {w:<6} {issue}")
    
    print(f"\nFinal spatial size before global pooling: {h}x{w}")
    
    if issues:
        print(f"\n{'='*70}")
        print("⚠️  POTENTIAL ISSUES DETECTED:")
        print(f"{'='*70}")
        for issue in issues:
            print(f"  • {issue}")
    else:
        print(f"\n✅ No critical dimension issues detected")
    
    return h, w, issues


def test_forward_pass(config, image_size=(29, 18)):
    """
    Actually build the model and test a forward pass.
    """
    print(f"\n{'='*70}")
    print(f"TESTING FORWARD PASS")
    print(f"{'='*70}")
    
    try:
        model = build_model(config)
        model.eval()
        
        h, w = image_size
        batch_size = 2
        x = torch.randn(batch_size, 1, h, w)
        
        print(f"Input tensor shape: {x.shape}")
        
        with torch.no_grad():
            output = model(x)
        
        print(f"Output tensor shape: {output.shape}")
        print(f"Expected output shape: [{batch_size}, {config['head']['num_classes']}]")
        
        if output.shape == (batch_size, config['head']['num_classes']):
            print(f"\n✅ Forward pass successful!")
            return True
        else:
            print(f"\n⚠️  Output shape mismatch!")
            return False
            
    except Exception as e:
        print(f"\n❌ Forward pass failed with error:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def compare_configurations():
    """
    Compare full SVHN vs cropped SVHN configurations.
    """
    print(f"\n{'='*70}")
    print(f"CONFIGURATION COMPARISON")
    print(f"{'='*70}")
    
    full_config = PRESETS['paper_svhn_rp']
    cropped_config = PRESETS['paper_svhn_rp_cropped']
    
    print(f"\nFull SVHN (paper_svhn_rp):")
    print(f"  Image size: {full_config['image_size']}")
    print(f"  Stages: {full_config['blocks']['stages']}")
    print(f"  Downsample at: {full_config['blocks']['downsample_at']}")
    
    print(f"\nCropped SVHN (paper_svhn_rp_cropped):")
    print(f"  Image size: {cropped_config['image_size']}")
    print(f"  Stages: {cropped_config['blocks']['stages']}")
    print(f"  Downsample at: {cropped_config['blocks']['downsample_at']}")
    
    # Check for differences
    if full_config['blocks']['downsample_at'] == cropped_config['blocks']['downsample_at']:
        print(f"\n⚠️  WARNING: Both configurations use the SAME downsampling schedule!")
        print(f"     This may be problematic for the smaller cropped images (29x18 vs 32x32).")
        print(f"\n💡 RECOMMENDATION:")
        print(f"     The cropped image is smaller (29x18 vs 32x32), so you may need to:")
        print(f"     1. Reduce the number of downsample operations")
        print(f"     2. Adjust which stages perform downsampling")
        print(f"     3. Or remove some downsample operations entirely")
    
    return full_config, cropped_config


def suggest_better_config(original_size=32, cropped_size=(29, 18)):
    """
    Suggest a better downsampling configuration for cropped images.
    """
    print(f"\n{'='*70}")
    print(f"SUGGESTED IMPROVEMENTS")
    print(f"{'='*70}")
    
    h, w = cropped_size
    min_dim = min(h, w)  # 18
    
    # Calculate maximum safe downsamples (keeping spatial size >= 2)
    max_downsamples = 0
    temp_dim = min_dim
    while temp_dim // 2 >= 2:
        max_downsamples += 1
        temp_dim = temp_dim // 2
    
    print(f"\nOriginal SVHN size: {original_size}x{original_size}")
    print(f"Cropped size: {h}x{w}")
    print(f"Minimum dimension: {min_dim}")
    print(f"Maximum safe downsamples (to keep dim >= 2): {max_downsamples}")
    
    # Original config has 4 downsamples at stages [1,3,5,7]
    # 32 -> 16 -> 8 -> 4 -> 2
    # But for 18: 18 -> 9 -> 4 -> 2 -> 1 (last one is too small!)
    
    print(f"\nOriginal config (paper_svhn_rp):")
    print(f"  Downsamples: 4 times at stages [1, 3, 5, 7]")
    print(f"  Spatial progression: 32 → 16 → 8 → 4 → 2")
    
    print(f"\nFor cropped images (29x18):")
    print(f"  Current: 18 → 9 → 4 → 2 → 1 ❌ (dimension 1 is too small!)")
    print(f"  Suggested: Use only {max_downsamples} downsamples")
    
    if max_downsamples == 3:
        print(f"  Better config: Downsample at stages [1, 3, 5] (remove stage 7)")
        print(f"  Spatial progression: 18 → 9 → 4 → 2 ✅")
    elif max_downsamples == 2:
        print(f"  Better config: Downsample at stages [1, 3] (remove stages 5, 7)")
        print(f"  Spatial progression: 18 → 9 → 4 ✅")
    
    print(f"\n💡 RECOMMENDED CHANGES TO 'paper_svhn_rp_cropped':")
    print(f"   Change: \"downsample_at\": [1, 3, 5, 7]")
    print(f"   To:     \"downsample_at\": [1, 3, 5]  (remove last downsample)")
    print(f"   Or:     \"downsample_at\": [1, 3]     (even safer)")


def main():
    print("="*70)
    print(" SVHN CROPPED IMAGE ARCHITECTURE DIAGNOSTICS")
    print("="*70)
    
    # Compare configs
    full_config, cropped_config = compare_configurations()
    
    # Check full SVHN
    print(f"\n\n{'#'*70}")
    print("# ANALYZING: paper_svhn_rp (FULL 32x32)")
    print(f"{'#'*70}")
    h1, w1, issues1 = check_spatial_dimensions(full_config, image_size=(32, 32))
    success1 = test_forward_pass(full_config, image_size=(32, 32))
    
    # Check cropped SVHN
    print(f"\n\n{'#'*70}")
    print("# ANALYZING: paper_svhn_rp_cropped (CROPPED 29x18)")
    print(f"{'#'*70}")
    h2, w2, issues2 = check_spatial_dimensions(cropped_config, image_size=(29, 18))
    success2 = test_forward_pass(cropped_config, image_size=(29, 18))
    
    # Provide suggestions
    suggest_better_config(original_size=32, cropped_size=(29, 18))
    
    # Final summary
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"Full SVHN (32x32): {'✅ OK' if success1 and not issues1 else '⚠️ Issues detected'}")
    print(f"Cropped SVHN (29x18): {'✅ OK' if success2 and not issues2 else '⚠️ Issues detected'}")
    
    if issues2:
        print(f"\n⚠️  The cropped configuration has {len(issues2)} issue(s).")
        print(f"    This likely explains the accuracy drop from 85% to 79%!")
        print(f"\n🔧 NEXT STEPS:")
        print(f"   1. Update the 'downsample_at' parameter in 'paper_svhn_rp_cropped'")
        print(f"   2. Retrain the model with the corrected configuration")
        print(f"   3. You should see better accuracy closer to or exceeding 85%")


if __name__ == "__main__":
    main()
