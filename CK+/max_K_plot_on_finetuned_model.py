import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from train_full import CKEmotionCNN

# ---------------------- Dataset / Model Metadata ----------------------
CLASS_NAMES = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise']
IMAGE_SIZE = 48
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
HEATMAP_DIR = os.path.join(os.path.dirname(__file__), "per_class_heatmaps")
DATA_DIR = os.path.join(os.path.dirname(__file__), "ck_dataset")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "CK_full_CNN_outputs", "masked_model_max_top40.pth")

# ---------------------- Load Per-Class Heatmaps ----------------------
heatmaps = []
for cls in CLASS_NAMES:
    path = os.path.join(HEATMAP_DIR, f"{cls}_heatmap.npy")
    heatmap = np.load(path)  # shape: (48, 48)
    heatmaps.append(heatmap)
global_heatmap = np.max(heatmaps, axis=0)

# Normalize global heatmap to [0, 1]
global_heatmap -= global_heatmap.min()
if global_heatmap.max() > 0:
    global_heatmap /= global_heatmap.max()

# ---------------------- Load Test Set ----------------------
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Group test images by class
class_to_images = {cls: [] for cls in CLASS_NAMES}
for cls in CLASS_NAMES:
    class_dir = os.path.join(DATA_DIR, cls)
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
            class_to_images[cls].append(img)

# Flatten all test images and labels
all_images = []
all_labels = []
for cls in CLASS_NAMES:
    all_images.extend(class_to_images[cls])
    all_labels.extend([CLASS_NAMES.index(cls)] * len(class_to_images[cls]))

if len(all_images) == 0:
    raise RuntimeError("No images loaded from CK+ dataset. Check DATA_DIR.")

all_images = torch.stack(all_images)
all_labels = torch.tensor(all_labels)

# ---------------------- Load Model ----------------------
print("\n🧠 Loading fine-tuned CK+ classifier (max 40%)...")
model = CKEmotionCNN(num_classes=len(CLASS_NAMES)).to(DEVICE)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# ---------------------- Evaluation Loop ----------------------
top_k_percents = list(range(20, 105, 5))  # [20, 25, ..., 100]
results = {k: {i: {'correct': 0, 'total': 0} for i in range(len(CLASS_NAMES))} for k in top_k_percents}

print("\nEvaluating for Top-K% values:", top_k_percents)

with torch.no_grad():
    batch_size = 128
    for i in range(0, len(all_images), batch_size):
        images = all_images[i:i+batch_size].to(DEVICE)
        labels = all_labels[i:i+batch_size].to(DEVICE)

        for k in top_k_percents:
            flat = global_heatmap.flatten()
            kth_threshold = np.percentile(flat, 100 - k)
            binary_mask = (global_heatmap >= kth_threshold).astype(np.float32)
            binary_mask_tensor = torch.tensor(binary_mask).unsqueeze(0).to(DEVICE)

            masked_images = images * binary_mask_tensor

            outputs = model(masked_images)
            _, predicted = torch.max(outputs, 1)

            for idx in range(len(labels)):
                label = labels[idx].item()
                pred = predicted[idx].item()
                results[k][label]['total'] += 1
                if label == pred:
                    results[k][label]['correct'] += 1

# ---------------------- Plotting ----------------------
plt.figure(figsize=(12, 6))
for idx, cls in enumerate(CLASS_NAMES):
    accuracies = [
        100 * results[k][idx]['correct'] / results[k][idx]['total'] if results[k][idx]['total'] > 0 else 0.0
        for k in top_k_percents
    ]
    plt.plot(top_k_percents, accuracies, marker='o', label=f"Class {cls}", linewidth=2, markersize=6)

# Plot average accuracy
avg_accuracies = []
for k in top_k_percents:
    total_correct = sum(results[k][idx]['correct'] for idx in range(len(CLASS_NAMES)))
    total_total = sum(results[k][idx]['total'] for idx in range(len(CLASS_NAMES)))
    avg_accuracies.append(100 * total_correct / total_total if total_total > 0 else 0.0)
plt.plot(top_k_percents, avg_accuracies, color='black', linestyle='--', linewidth=3, marker='o', markersize=6, label='Average Accuracy')

plt.title("CK+ Per-Class Accuracy vs Top-K% on Fine-tuned Model (Max 40%)", fontsize=16, fontweight='bold')
plt.xlabel("Top-K% Pixels Kept", fontsize=14)
plt.ylabel("Accuracy (%)", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)
plt.tight_layout()
output_plot_path = os.path.join(os.path.dirname(__file__), "global_mask_topk_max_vs_accuracy_on_finetuned40.png")
plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
print(f"\nPlot saved to: {output_plot_path}")
plt.close()
