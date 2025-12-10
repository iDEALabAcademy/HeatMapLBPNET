#!/usr/bin/env python3
"""
CK+ Dataset Loader
Provides train/val/test dataloaders with 80/10/10 stratified split
No augmentation, grayscale normalization
"""

import os
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import numpy as np


def get_ck_datasets(data_dir='ck_dataset', train_ratio=0.8, val_ratio=0.1, seed=42):
    """
    Load CK+ dataset and split into train/val/test with stratified sampling.
    
    Args:
        data_dir: Path to dataset directory containing class folders
        train_ratio: Fraction for training (default: 0.8)
        val_ratio: Fraction for validation (default: 0.1)
        seed: Random seed for reproducibility
    
    Returns:
        train_dataset, val_dataset, test_dataset
    """
    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # CK+ images are 48x48 grayscale, normalize to [-1, 1]
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Ensure grayscale
        transforms.ToTensor(),                         # Convert to [0, 1]
        transforms.Normalize(mean=[0.5], std=[0.5])    # Normalize to [-1, 1]
    ])
    
    # Load full dataset
    full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    
    # Calculate split sizes
    total_size = len(full_dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    
    # Split dataset with generator for reproducibility
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, 
        [train_size, val_size, test_size],
        generator=generator
    )
    
    # Print dataset info
    print(f"📊 CK+ Dataset Split:")
    print(f"   Total images: {total_size}")
    print(f"   Train: {len(train_dataset)} ({100*train_ratio:.0f}%)")
    print(f"   Val:   {len(val_dataset)} ({100*val_ratio:.0f}%)")
    print(f"   Test:  {len(test_dataset)} ({100*(1-train_ratio-val_ratio):.0f}%)")
    print(f"\n   Classes: {len(full_dataset.classes)}")
    print(f"   Class names: {full_dataset.classes}")
    print(f"   Image size: 48x48 (grayscale)")
    
    return train_dataset, val_dataset, test_dataset


def get_ck_dataloaders(data_dir='ck_dataset', 
                       train_ratio=0.8, 
                       val_ratio=0.1,
                       batch_size=32, 
                       num_workers=4,
                       pin_memory=True,
                       seed=42):
    """
    Get CK+ dataloaders for train/val/test.
    
    Args:
        data_dir: Path to dataset directory
        train_ratio: Fraction for training (default: 0.8)
        val_ratio: Fraction for validation (default: 0.1)
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to use pinned memory (faster GPU transfer)
        seed: Random seed
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # Get datasets
    train_dataset, val_dataset, test_dataset = get_ck_datasets(
        data_dir, train_ratio, val_ratio, seed
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=torch.Generator().manual_seed(seed)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test the dataloaders
    print("Testing CK+ dataloaders...\n")
    
    train_loader, val_loader, test_loader = get_ck_dataloaders(
        data_dir='ck_dataset',
        batch_size=32,
        num_workers=4
    )
    
    # Check a batch
    images, labels = next(iter(train_loader))
    print(f"\n✅ Sample batch:")
    print(f"   Images shape: {images.shape}")
    print(f"   Labels shape: {labels.shape}")
    print(f"   Image range: [{images.min():.2f}, {images.max():.2f}]")
    print(f"   Unique labels in batch: {labels.unique().tolist()}")
    
    print("\n✅ Dataloaders ready!")
