#!/usr/bin/env python3
import os
import sys
import argparse
import math
import torch
import numpy as np
import matplotlib.pyplot as plt

# import from repo root
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from train_original_model import get_config  # noqa: E402
from lbpnet.models import build_model  # noqa: E402
from lbpnet.data import get_mnist_datasets  # noqa: E402


def unnormalize(img_tensor: torch.Tensor) -> torch.Tensor:
    # MNIST normalization: mean=0.1307, std=0.3081
    mean, std = 0.1307, 0.3081
    return img_tensor * std + mean


@torch.no_grad()
def collect_predictions(model: torch.nn.Module, loader, device: torch.device):
    model.eval()
    if hasattr(model, 'set_ste'):
        model.set_ste(True, True)
    all_logits = []
    all_targets = []
    all_images = []
    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        logits = model(images)
        all_logits.append(logits.detach().cpu())
        all_targets.append(targets.detach().cpu())
        all_images.append(images.detach().cpu())
    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)
    images = torch.cat(all_images, dim=0)
    preds = logits.argmax(dim=1)
    return images, logits, preds, targets


def plot_confusion_matrix(cm: np.ndarray, classes: list[str], save_path: str):
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           ylabel='True label', xlabel='Predicted label', title='MNIST Confusion Matrix')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    # write normalized values
    cm_sum = cm.sum(axis=1, keepdims=True)
    cm_norm = cm / np.maximum(cm_sum, 1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}\n({cm_norm[i, j]*100:.1f}%)",
                    ha='center', va='center', color='white' if cm[i, j] > cm.max()*0.6 else 'black', fontsize=7)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


def plot_pca_scatter(logits: torch.Tensor, targets: torch.Tensor, save_path: str, max_points: int = 3000):
    X = logits.detach().cpu().numpy()
    y = targets.detach().cpu().numpy()
    if X.shape[0] > max_points:
        idx = np.random.RandomState(42).choice(X.shape[0], size=max_points, replace=False)
        X = X[idx]
        y = y[idx]
    # PCA via SVD
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    comps = Vt[:2]  # (2, C)
    Z = Xc @ comps.T  # (N, 2)
    # plot
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    scatter = ax.scatter(Z[:, 0], Z[:, 1], c=y, s=6, cmap='tab10', alpha=0.8)
    legend1 = ax.legend(*scatter.legend_elements(), title='Classes', loc='best', fontsize=8)
    ax.add_artist(legend1)
    ax.set_title('MNIST PCA (from logits)')
    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


def make_grid(images: torch.Tensor, titles: list[str], ncols: int, save_path: str):
    n = images.size(0)
    ncols = max(1, ncols)
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.5, nrows * 1.5), dpi=150)
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = np.array([axes])
    elif ncols == 1:
        axes = np.array([[ax] for ax in axes])
    for i in range(nrows * ncols):
        r = i // ncols
        c = i % ncols
        ax = axes[r, c]
        ax.axis('off')
        if i < n:
            img = images[i, 0, :, :].numpy()
            ax.imshow(img, cmap='gray')
            if i < len(titles):
                ax.set_title(titles[i], fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--preset', type=str, default=os.environ.get('MODEL_PRESET', 'paper_mnist_rp'))
    ap.add_argument('--ckpt', type=str, default=os.path.join('./outputs_mnist_original', 'best_model.pth'))
    ap.add_argument('--limit', type=int, default=None, help='limit number of test samples for speed')
    ap.add_argument('--outdir', type=str, default=os.path.join('./outputs_mnist_original', 'viz'))
    args = ap.parse_args()

    os.environ['MODEL_PRESET'] = args.preset
    cfg = get_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # datasets and loader
    train_ds, val_ds, test_ds = get_mnist_datasets(cfg['data'], download=False)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)

    # build model
    model = build_model(cfg).to(device)
    # dummy forward to init dynamic buffers
    model.eval()
    with torch.no_grad():
        H = W = int(cfg.get('image_size', 28))
        _ = model(torch.zeros(8, 1, H, W, device=device))
    # load checkpoint
    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f'Checkpoint not found: {args.ckpt}')
    state = torch.load(args.ckpt, map_location=device)
    raw = state.get('model_state_dict', state)
    model.load_state_dict(raw, strict=False)
    if hasattr(model, 'set_ste'):
        model.set_ste(True, True)

    # collect predictions
    images, logits, preds, targets = collect_predictions(model, test_loader, device)
    if args.limit is not None and images.size(0) > args.limit:
        images = images[:args.limit]
        logits = logits[:args.limit]
        preds = preds[:args.limit]
        targets = targets[:args.limit]

    # ensure outdir
    os.makedirs(args.outdir, exist_ok=True)

    # confusion matrix
    num_classes = int(cfg['head'].get('num_classes', 10))
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(targets.numpy().tolist(), preds.numpy().tolist()):
        cm[t, p] += 1
    plot_confusion_matrix(cm, classes=[str(i) for i in range(num_classes)],
                          save_path=os.path.join(args.outdir, 'mnist_confusion_matrix.png'))

    # PCA scatter from logits
    plot_pca_scatter(logits, targets, save_path=os.path.join(args.outdir, 'mnist_pca_logits.png'))

    # sample grids: correct and incorrect
    correct_idx = (preds == targets).nonzero(as_tuple=False).squeeze(1)
    wrong_idx = (preds != targets).nonzero(as_tuple=False).squeeze(1)
    # take first 40 of each
    correct_idx = correct_idx[:40]
    wrong_idx = wrong_idx[:40]
    # unnormalize for visualization
    imgs_correct = unnormalize(images[correct_idx]).clamp(0, 1)
    imgs_wrong = unnormalize(images[wrong_idx]).clamp(0, 1)
    titles_correct = [f"y={int(targets[i])}, p={int(preds[i])}" for i in correct_idx.tolist()]
    titles_wrong = [f"y={int(targets[i])}, p={int(preds[i])}" for i in wrong_idx.tolist()]
    make_grid(imgs_correct, titles_correct, ncols=10, save_path=os.path.join(args.outdir, 'mnist_correct_grid.png'))
    make_grid(imgs_wrong, titles_wrong, ncols=10, save_path=os.path.join(args.outdir, 'mnist_wrong_grid.png'))

    # class-wise accuracy bar (optional)
    cls_total = cm.sum(axis=1)
    cls_acc = np.divide(np.diag(cm), np.maximum(cls_total, 1))
    fig, ax = plt.subplots(figsize=(7, 3), dpi=150)
    ax.bar(np.arange(num_classes), cls_acc * 100.0)
    ax.set_title('MNIST Class-wise Accuracy')
    ax.set_xlabel('Class')
    ax.set_ylabel('Accuracy (%)')
    ax.set_xticks(np.arange(num_classes))
    ax.set_xticklabels([str(i) for i in range(num_classes)])
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, 'mnist_classwise_accuracy.png'), bbox_inches='tight')
    plt.close(fig)

    print(f"Saved visualizations to: {os.path.abspath(args.outdir)}")


if __name__ == '__main__':
    main()









