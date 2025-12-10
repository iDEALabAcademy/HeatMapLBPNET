#!/usr/bin/env python3
"""
Train a CNN on CK+ Facial Expression Dataset
- Dataset: 981 images, 48x48 grayscale, 7 emotion classes
- Split: 80/10/10 train/val/test
- Architecture: Lightweight CNN appropriate for small dataset
- Saves: model, plots, training logs
"""

import os
import time
import csv
import torch
import torch.nn as nn
import torch.optim as optim
try:
    from torch.amp import autocast, GradScaler
except ImportError:
    from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt
import numpy as np

from data_loader import get_ck_dataloaders


class CKEmotionCNN(nn.Module):
    """
    Lightweight CNN for CK+ (48x48 grayscale, 7 classes)
    Designed to avoid overfitting on small dataset (~785 training images)
    """
    def __init__(self, num_classes=7, dropout_rate=0.6):
        super(CKEmotionCNN, self).__init__()
        
        # Feature extraction - REDUCED CAPACITY
        self.features = nn.Sequential(
            # Conv1: 48x48 -> 24x24
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # Reduced: 32->16
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),
            
            # Conv2: 24x24 -> 12x12
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # Reduced: 64->32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.3),
            
            # Conv3: 12x12 -> 6x6
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # Reduced: 128->64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.4),
        )
        
        # Classifier - REDUCED CAPACITY
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 6 * 6, 128),  # Reduced: 128*6*6->256 now 64*6*6->128
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)  # Reduced: 256->128
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def train_epoch(model, train_loader, criterion, optimizer, scaler, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        with autocast(device_type='cuda' if device.type == 'cuda' else 'cpu', enabled=(device.type == 'cuda')):
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        # Backward pass
        if device.type == 'cuda':
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        # Statistics
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            
            with autocast(device_type='cuda' if device.type == 'cuda' else 'cpu', enabled=(device.type == 'cuda')):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def test_per_class(model, test_loader, device, class_names):
    """Test and return per-class accuracy"""
    model.eval()
    class_correct = [0] * len(class_names)
    class_total = [0] * len(class_names)
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            with autocast(device_type='cuda' if device.type == 'cuda' else 'cpu', enabled=(device.type == 'cuda')):
                outputs = model(images)
            
            _, predicted = outputs.max(1)
            
            for label, pred in zip(labels, predicted):
                class_total[label] += 1
                if label == pred:
                    class_correct[label] += 1
    
    # Calculate per-class accuracy
    class_accuracies = []
    for i in range(len(class_names)):
        if class_total[i] > 0:
            acc = 100.0 * class_correct[i] / class_total[i]
            class_accuracies.append(acc)
        else:
            class_accuracies.append(0.0)
    
    return class_correct, class_total, class_accuracies


def save_plots(train_losses, val_losses, train_accs, val_accs, output_dir):
    """Save training curves"""
    epochs = range(1, len(train_losses) + 1)
    
    # Loss plot
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Train Loss')
    plt.plot(epochs, val_losses, 'r-', label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, 'b-', label='Train Acc')
    plt.plot(epochs, val_accs, 'r-', label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=150)
    plt.close()
    print(f"✅ Saved training curves to {output_dir}/training_curves.png")


def train_model():
    """Main training function"""
    # Configuration
    config = {
        'batch_size': 32,
        'num_epochs': 100,
        'learning_rate': 5e-4,  # Lower LR
        'weight_decay': 5e-4,   # Stronger weight decay
        'num_workers': 4,
        'patience': 20,  # Early stopping patience
        'seed': 42,
        'dropout_rate': 0.6,  # Increase dropout
        'label_smoothing': 0.1  # Add label smoothing
    }
    
    # Setup
    output_dir = 'CK_full_CNN_outputs'
    os.makedirs(output_dir, exist_ok=True)
    
    # Set seeds
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['seed'])
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    # Data loaders
    print("\n📂 Loading CK+ dataset...")
    train_loader, val_loader, test_loader = get_ck_dataloaders(
        data_dir='ck_dataset',
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        seed=config['seed']
    )
    
    # Model
    print(f"\n🏗️  Building model...")
    model = CKEmotionCNN(num_classes=7, dropout_rate=config['dropout_rate']).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=config.get('label_smoothing', 0.0))
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
    scaler = GradScaler() if device.type == 'cuda' else None
    
    # Training tracking
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0.0
    patience_counter = 0
    
    # CSV log
    csv_path = os.path.join(output_dir, 'training_log.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Train_Loss', 'Train_Acc', 'Val_Loss', 'Val_Acc', 'LR', 'Time(s)'])
    
    # Training loop
    print(f"\n🎯 Starting training for {config['num_epochs']} epochs...")
    print("=" * 80)
    
    for epoch in range(1, config['num_epochs'] + 1):
        epoch_start = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler, device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step(val_acc)
        
        # Track metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # Time
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log to CSV
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f'{train_loss:.4f}', f'{train_acc:.2f}', 
                           f'{val_loss:.4f}', f'{val_acc:.2f}', f'{current_lr:.6f}', f'{epoch_time:.2f}'])
        
        # Print progress
        print(f"Epoch [{epoch:3d}/{config['num_epochs']}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
              f"LR: {current_lr:.6f} | Time: {epoch_time:.1f}s")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'config': config
            }, os.path.join(output_dir, 'best_model.pth'))
            print(f"   ✅ New best validation accuracy: {val_acc:.2f}%")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config['patience']:
            print(f"\n⚠️  Early stopping triggered after {epoch} epochs (patience={config['patience']})")
            break
    
    print("=" * 80)
    print(f"✅ Training complete! Best validation accuracy: {best_val_acc:.2f}%")
    
    # Save plots
    save_plots(train_losses, val_losses, train_accs, val_accs, output_dir)
    
    # Test evaluation
    print("\n📊 Evaluating on test set...")
    checkpoint = torch.load(os.path.join(output_dir, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"   Test Loss: {test_loss:.4f}")
    print(f"   Test Accuracy: {test_acc:.2f}%")
    
    # Per-class test accuracy
    print("\n📈 Per-class Test Accuracy:")
    print("=" * 60)
    
    # Get class names from dataset
    # Access the underlying dataset from the Subset
    if hasattr(test_loader.dataset, 'dataset'):
        class_names = test_loader.dataset.dataset.classes
    else:
        class_names = test_loader.dataset.classes
    
    class_correct, class_total, class_accuracies = test_per_class(model, test_loader, device, class_names)
    
    for i, class_name in enumerate(class_names):
        print(f"   {class_name:15s}: {class_correct[i]:3d}/{class_total[i]:3d} = {class_accuracies[i]:6.2f}%")
    
    print("=" * 60)
    print(f"   Overall:        {sum(class_correct):3d}/{sum(class_total):3d} = {test_acc:6.2f}%")
    
    # Save test results
    with open(os.path.join(output_dir, 'test_results.txt'), 'w') as f:
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Test Accuracy: {test_acc:.2f}%\n")
        f.write(f"Best Val Accuracy: {best_val_acc:.2f}%\n\n")
        f.write("Per-class Test Accuracy:\n")
        f.write("=" * 60 + "\n")
        for i, class_name in enumerate(class_names):
            f.write(f"{class_name:15s}: {class_correct[i]:3d}/{class_total[i]:3d} = {class_accuracies[i]:6.2f}%\n")
        f.write("=" * 60 + "\n")
        f.write(f"Overall:        {sum(class_correct):3d}/{sum(class_total):3d} = {test_acc:6.2f}%\n")
    
    print(f"\n✅ All outputs saved to {output_dir}/")


if __name__ == "__main__":
    train_model()
