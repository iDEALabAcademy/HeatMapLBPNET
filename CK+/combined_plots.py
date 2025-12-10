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
MODEL_PATH_BASE = os.path.join(os.path.dirname(__file__), "CK_full_CNN_outputs", "best_model.pth")
MODEL_PATH_AVG_FT = os.path.join(os.path.dirname(__file__), "CK_full_CNN_outputs", "masked_model_avg_top40.pth")
MODEL_PATH_MAX_FT = os.path.join(os.path.dirname(__file__), "CK_full_CNN_outputs", "masked_model_max_top40.pth")

# ---------------------- Load Per-Class Heatmaps ----------------------
heatmaps = []
for cls in CLASS_NAMES:
    path = os.path.join(HEATMAP_DIR, f"{cls}_heatmap.npy")
    heatmap = np.load(path)  # shape: (48, 48)
    heatmaps.append(heatmap)

global_heatmap_avg = np.mean(heatmaps, axis=0)
global_heatmap_max = np.max(heatmaps, axis=0)

# Normalize global heatmaps to [0, 1]
global_heatmap_avg -= global_heatmap_avg.min()
if global_heatmap_avg.max() > 0:
    global_heatmap_avg /= global_heatmap_avg.max()

global_heatmap_max -= global_heatmap_max.min()
if global_heatmap_max.max() > 0:
    global_heatmap_max /= global_heatmap_max.max()

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

top_k_percents = list(range(20, 105, 5))  # [20, 25, ..., 100]

# ---------------------- Helper Function ----------------------
def evaluate_model(model_path, global_heatmap, title_prefix):
    """Evaluate a model with a given global heatmap"""
    print(f"\n🧠 Loading {title_prefix}...")
    model = CKEmotionCNN(num_classes=len(CLASS_NAMES)).to(DEVICE)
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    results = {k: {i: {'correct': 0, 'total': 0} for i in range(len(CLASS_NAMES))} for k in top_k_percents}
    
    print(f"Evaluating {title_prefix}...")
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
    
    return results

# ---------------------- Evaluate All 4 Configurations ----------------------
print("=" * 70)
print("Evaluating all 4 configurations...")
print("=" * 70)

results_avg = evaluate_model(MODEL_PATH_BASE, global_heatmap_avg, "Average-based (Base Model)")
results_max = evaluate_model(MODEL_PATH_BASE, global_heatmap_max, "Max-based (Base Model)")
results_avg_ft = evaluate_model(MODEL_PATH_AVG_FT, global_heatmap_avg, "Average-based (Fine-tuned 40%)")
results_max_ft = evaluate_model(MODEL_PATH_MAX_FT, global_heatmap_max, "Max-based (Fine-tuned 40%)")

# ---------------------- Plotting ----------------------
fig, axes = plt.subplots(2, 2, figsize=(20, 16))

configurations = [
    (results_avg, "Average-based", axes[0, 0]),
    (results_max, "Max-based", axes[0, 1]),
    (results_avg_ft, "Average-based (Fine-tuned 40%)", axes[1, 0]),
    (results_max_ft, "Max-based (Fine-tuned 40%)", axes[1, 1])
]

# Color scheme for consistent colors across plots
colors = plt.cm.tab10(range(len(CLASS_NAMES)))

for results, title, ax in configurations:
    # Plot per-class accuracies
    for idx, cls in enumerate(CLASS_NAMES):
        accuracies = [
            100 * results[k][idx]['correct'] / results[k][idx]['total'] if results[k][idx]['total'] > 0 else 0.0
            for k in top_k_percents
        ]
        ax.plot(top_k_percents, accuracies, marker='o', label=f"{cls}", 
                linewidth=2.5, markersize=7, color=colors[idx])
    
    # Plot average accuracy
    avg_accuracies = []
    for k in top_k_percents:
        total_correct = sum(results[k][idx]['correct'] for idx in range(len(CLASS_NAMES)))
        total_total = sum(results[k][idx]['total'] for idx in range(len(CLASS_NAMES)))
        avg_accuracies.append(100 * total_correct / total_total if total_total > 0 else 0.0)
    ax.plot(top_k_percents, avg_accuracies, color='black', linestyle='--', 
            linewidth=3.5, marker='o', markersize=8, label='Average')
    
    ax.set_title(title, fontsize=20, fontweight='bold', pad=15)
    ax.set_xlabel("Top-K% Pixels Kept", fontsize=18, fontweight='bold')
    ax.set_ylabel("Accuracy (%)", fontsize=18, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.grid(True, alpha=0.3, linewidth=1.2)

plt.tight_layout()
output_plot_path = os.path.join(os.path.dirname(__file__), "combined_topk_accuracy_plots.png")
plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
print(f"\n{'='*70}")
print(f"✓ Combined plot saved to: {output_plot_path}")

# Create separate legend figure
handles, labels = axes[0, 0].get_legend_handles_labels()
fig_legend = plt.figure(figsize=(16, 1.5))
fig_legend.legend(handles, labels, loc='center', ncol=8, fontsize=16, 
                 frameon=True, fancybox=True, shadow=True)
legend_path = os.path.join(os.path.dirname(__file__), "combined_topk_accuracy_legend.png")
fig_legend.savefig(legend_path, dpi=300, bbox_inches='tight')
print(f"✓ Legend saved to: {legend_path}")
print(f"{'='*70}")
plt.close('all')
