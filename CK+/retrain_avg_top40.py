import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from train_full import CKEmotionCNN

# ---------------------- Configuration ----------------------
MASK_PATH = "global_heatmap_avg/global_avg_binary_mask.npy"
PRETRAINED_MODEL_PATH = "CK_full_CNN_outputs/best_model.pth"
MODEL_SAVE_PATH = "CK_full_CNN_outputs/masked_model_avg_top40.pth"
DATA_DIR = "ck_dataset"
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 1e-4
CLASS_NAMES = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------------- Load Global Binary Mask ----------------------
binary_mask = np.load(MASK_PATH)  # shape: (48, 48)
binary_mask_tensor = torch.tensor(binary_mask).unsqueeze(0).to(device)  # shape: (1, 48, 48)

# ---------------------- Custom Masked Dataset Wrapper ----------------------
class MaskedCKPlus(torch.utils.data.Dataset):
    def __init__(self, images, labels, binary_mask_tensor):
        self.images = images
        self.labels = labels
        self.mask = binary_mask_tensor  # shape: (1, 48, 48)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        masked_img = img * self.mask.cpu()
        return masked_img, label

# ---------------------- Data Preparation ----------------------
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load all images from dataset
all_images = []
all_labels = []

for cls_idx, cls_name in enumerate(CLASS_NAMES):
    class_dir = os.path.join(DATA_DIR, cls_name)
    if not os.path.isdir(class_dir):
        print(f"⚠️  Warning: missing class directory {class_dir}")
        continue
    for fname in os.listdir(class_dir):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(class_dir, fname)
            img = plt.imread(img_path)
            if img.max() > 1.0:
                img = img / 255.0
            img = transforms.ToPILImage()(img) if isinstance(img, np.ndarray) else img
            img = transform(img)
            all_images.append(img)
            all_labels.append(cls_idx)

all_images = torch.stack(all_images)
all_labels = torch.tensor(all_labels)

# Split into train/test (80/20)
from sklearn.model_selection import train_test_split
train_images, test_images, train_labels, test_labels = train_test_split(
    all_images, all_labels, test_size=0.2, random_state=42, stratify=all_labels
)

train_dataset = MaskedCKPlus(train_images, train_labels, binary_mask_tensor)
test_dataset = MaskedCKPlus(test_images, test_labels, binary_mask_tensor)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ---------------------- Load and Fine-Tune Model ----------------------
model = CKEmotionCNN(num_classes=len(CLASS_NAMES)).to(device)
checkpoint = torch.load(PRETRAINED_MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

# Unfreeze all layers for full fine-tuning
for param in model.parameters():
    param.requires_grad = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ---------------------- Training Loop ----------------------
print("Fine-tuning on avg-based masked dataset (40%)...")
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%")

# ---------------------- Save Fine-Tuned Model ----------------------
torch.save({'model_state_dict': model.state_dict()}, MODEL_SAVE_PATH)
print(f"\nFine-tuned model saved to: {MODEL_SAVE_PATH}")

# ---------------------- Evaluate on Test Set (Per-Class Accuracy) ----------------------
print("\nEvaluating on masked test set...")
model.eval()
total_correct = 0
total_samples = 0
class_correct = {i: 0 for i in range(len(CLASS_NAMES))}
class_total = {i: 0 for i in range(len(CLASS_NAMES))}

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)

        for i in range(len(labels)):
            label = labels[i].item()
            pred = predicted[i].item()
            class_total[label] += 1
            if pred == label:
                class_correct[label] += 1

acc = 100 * total_correct / total_samples
print(f"\nOverall Test Accuracy: {acc:.2f}%")

print("\n--- Per-Class Accuracy ---")
for cls in range(len(CLASS_NAMES)):
    acc_cls = 100 * class_correct[cls] / class_total[cls] if class_total[cls] > 0 else 0.0
    print(f"{CLASS_NAMES[cls]}: {acc_cls:.2f}%")

# ---------------------- Save Sample Masked Images ----------------------
print("\nSaving 1 masked image from train and test set for verification...")
sample_output_dir = "sample_masked_avg_top40_for_retrain"
os.makedirs(sample_output_dir, exist_ok=True)

train_idx = random.randint(0, len(train_images)-1)
test_idx = random.randint(0, len(test_images)-1)
train_img_orig = train_images[train_idx]
test_img_orig = test_images[test_idx]
train_img_masked, _ = train_dataset[train_idx]
test_img_masked, _ = test_dataset[test_idx]

# Save images
def save_image(tensor, title, filename):
    plt.figure(figsize=(6, 6))
    plt.imshow(tensor.squeeze().numpy(), cmap='gray')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.savefig(os.path.join(sample_output_dir, filename), bbox_inches='tight', dpi=300)
    plt.close()

save_image(train_img_orig, "Train Original", "train_original_40.png")
save_image(train_img_masked, "Train Masked", "train_masked_40.png")
save_image(test_img_orig, "Test Original", "test_original_40.png")
save_image(test_img_masked, "Test Masked", "test_masked_40.png")

print(f"Saved to '{sample_output_dir}/'")
