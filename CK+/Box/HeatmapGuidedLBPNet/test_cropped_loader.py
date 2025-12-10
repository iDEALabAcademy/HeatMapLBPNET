#!/usr/bin/env python3
"""
Test script to verify SVHN cropped data loading.
Loads 1 random image per class from both train and test sets.
Saves them to sample_images/ directory.
"""

import os
import sys
import numpy as np
import torch
from PIL import Image
import yaml

# Add parent directory to path if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lbpnet.data import get_svhn_dataloaders


def denormalize(tensor, mean=0.5, std=0.5):
    """Convert from [-1, 1] back to [0, 1] for visualization."""
    return tensor * std + mean


def save_samples_per_class(loader, split_name, output_dir, num_classes=10):
    """
    Extract and save one random sample per class from the loader.
    
    Args:
        loader: DataLoader to extract samples from
        split_name: 'train' or 'test' for labeling
        output_dir: Directory to save images
        num_classes: Number of classes (default 10 for SVHN)
    """
    print(f"\n{'='*60}")
    print(f"Processing {split_name.upper()} set")
    print(f"{'='*60}")
    
    # Dictionary to store first occurrence of each class
    class_samples = {}
    
    # Iterate through loader to find one sample per class
    for batch_idx, (images, labels) in enumerate(loader):
        for i in range(len(labels)):
            label = int(labels[i].item())
            
            # If we haven't seen this class yet, save it
            if label not in class_samples:
                class_samples[label] = images[i]
                print(f"  Found class {label} in batch {batch_idx}")
                
            # Stop if we have all classes
            if len(class_samples) == num_classes:
                break
        
        if len(class_samples) == num_classes:
            break
    
    # Save the collected samples
    print(f"\nSaving {len(class_samples)} samples from {split_name} set...")
    for label, img_tensor in class_samples.items():
        # img_tensor shape: [1, H, W] (grayscale)
        img_denorm = denormalize(img_tensor)  # [-1,1] -> [0,1]
        img_np = (img_denorm.squeeze(0).numpy() * 255).astype(np.uint8)
        
        # Get dimensions
        h, w = img_np.shape
        
        # Save image
        filename = f"{split_name}_class_{label}_size_{h}x{w}.png"
        filepath = os.path.join(output_dir, filename)
        Image.fromarray(img_np, mode='L').save(filepath)
        print(f"  Saved: {filename} (shape: {h}×{w})")
    
    # Report if any classes were missing
    missing = set(range(num_classes)) - set(class_samples.keys())
    if missing:
        print(f"\n⚠️  WARNING: Missing classes in {split_name} set: {sorted(missing)}")
    else:
        print(f"\n✅ All {num_classes} classes found in {split_name} set!")
    
    return class_samples


def main():
    # Output directory
    output_dir = "sample_images"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {os.path.abspath(output_dir)}")
    
    # Load configuration (minimal config for testing)
    config = {
        'seed': 42,
        'data': {
            'train_ratio': 0.85,
            'val_size': -1,  # Use ratio-based split
            'stratified': True,
            'local_root': '/home/sgram/Heatmap/SVHN/SVHN_Data_grayscale',
            'bbox_path': '/home/sgram/Heatmap/SVHN/Box/avg_top40_SVHN/bbox.npy',
            'normalize': True,  # [-1, 1] normalization
            'num_workers': 0,
            'pin_memory': False
        },
        'training': {
            'batch_size': 64,
            'num_workers': 0,
            'pin_memory': False
        }
    }
    
    # Load bbox and verify dimensions
    bbox_path = config['data']['bbox_path']
    bbox = np.load(bbox_path).astype(int)
    t, l, b, r = bbox
    h_crop = b - t + 1
    w_crop = r - l + 1
    
    print(f"\n{'='*60}")
    print(f"BBOX INFORMATION")
    print(f"{'='*60}")
    print(f"BBox path: {bbox_path}")
    print(f"BBox values: [top={t}, left={l}, bottom={b}, right={r}]")
    print(f"Cropped size: H={h_crop}, W={w_crop}")
    print(f"Expected size: H=29, W=18")
    if h_crop == 29 and w_crop == 18:
        print("✅ Dimensions match expected values!")
    else:
        print(f"⚠️  WARNING: Dimensions don't match! Got {h_crop}×{w_crop}, expected 29×18")
    
    # Get dataloaders
    print(f"\n{'='*60}")
    print(f"LOADING DATALOADERS")
    print(f"{'='*60}")
    train_loader, val_loader, test_loader = get_svhn_dataloaders(
        config,
        data_dir=config['data']['local_root'],
        download=False
    )
    
    print(f"\nDataLoader information:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader) if val_loader else 0}")
    print(f"  Test batches: {len(test_loader)}")
    
    # Test a single batch to verify shape
    print(f"\n{'='*60}")
    print(f"VERIFYING BATCH SHAPES")
    print(f"{'='*60}")
    
    train_batch = next(iter(train_loader))
    test_batch = next(iter(test_loader))
    
    train_imgs, train_labels = train_batch
    test_imgs, test_labels = test_batch
    
    print(f"\nTrain batch:")
    print(f"  Images shape: {train_imgs.shape}  (expected: [batch, 1, 29, 18])")
    print(f"  Labels shape: {train_labels.shape}")
    print(f"  Image dtype: {train_imgs.dtype}")
    print(f"  Image range: [{train_imgs.min():.3f}, {train_imgs.max():.3f}]  (expected: ~[-1, 1])")
    
    print(f"\nTest batch:")
    print(f"  Images shape: {test_imgs.shape}  (expected: [batch, 1, 29, 18])")
    print(f"  Labels shape: {test_labels.shape}")
    print(f"  Image dtype: {test_imgs.dtype}")
    print(f"  Image range: [{test_imgs.min():.3f}, {test_imgs.max():.3f}]  (expected: ~[-1, 1])")
    
    # Verify dimensions
    _, C, H, W = train_imgs.shape
    if C == 1 and H == 29 and W == 18:
        print(f"\n✅ Train images have correct shape: [batch, 1, 29, 18]")
    else:
        print(f"\n⚠️  WARNING: Train images have unexpected shape: [batch, {C}, {H}, {W}]")
    
    _, C, H, W = test_imgs.shape
    if C == 1 and H == 29 and W == 18:
        print(f"✅ Test images have correct shape: [batch, 1, 29, 18]")
    else:
        print(f"⚠️  WARNING: Test images have unexpected shape: [batch, {C}, {H}, {W}]")
    
    # Verify normalization
    if -1.1 < train_imgs.min() < -0.9 and 0.9 < train_imgs.max() < 1.1:
        print(f"✅ Images appear to be normalized to [-1, 1]")
    else:
        print(f"⚠️  WARNING: Images may not be properly normalized")
    
    # Save samples
    print(f"\n{'='*60}")
    print(f"EXTRACTING AND SAVING SAMPLES")
    print(f"{'='*60}")
    
    train_samples = save_samples_per_class(train_loader, "train", output_dir)
    test_samples = save_samples_per_class(test_loader, "test", output_dir)
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Saved {len(train_samples)} train samples")
    print(f"✅ Saved {len(test_samples)} test samples")
    print(f"📁 Output directory: {os.path.abspath(output_dir)}")
    print(f"\nAll images saved with format: <split>_class_<label>_size_<H>x<W>.png")
    print(f"\nVerification complete!")


if __name__ == "__main__":
    main()
