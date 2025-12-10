#!/usr/bin/env python3
import os
import sys
import argparse
import torch

# 允许从仓库根目录导入 lbp 相关模块
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from train_original_model import get_config  # noqa: E402
from lbpnet.models import build_model  # noqa: E402
from SVHN.Box.binary_Ding.lbpnet.data.svhn_dataset import get_mnist_dataloaders  # noqa: E402


def evaluate_accuracy(model: torch.nn.Module, loader, device: torch.device) -> float:
    model.eval()
    if hasattr(model, 'set_ste'):
        # 确保硬路径前向
        model.set_ste(True, True)
    correct = 0
    total = 0
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1)
            correct += int((preds == targets).sum())
            total += int(targets.numel())
    return 100.0 * correct / max(total, 1)


def bn_recalibrate(model: torch.nn.Module, train_loader, device: torch.device, max_batches: int = 200) -> None:
    # 使用训练分布，开启训练态以刷新 BN 统计；同时保持硬路径
    model.train()
    if hasattr(model, 'set_ste'):
        model.set_ste(True, True)
    seen = 0
    with torch.no_grad():
        for images, _ in train_loader:
            images = images.to(device)
            _ = model(images)
            seen += 1
            if seen >= max_batches:
                break
    model.eval()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--preset', type=str, default=os.environ.get('MODEL_PRESET', 'paper_mnist_rp'))
    parser.add_argument('--ckpt', type=str, required=True, help='路径到 best_model.pth 或其他权重')
    parser.add_argument('--max-batches', type=int, default=200)
    args = parser.parse_args()

    os.environ['MODEL_PRESET'] = args.preset
    cfg = get_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 采用与训练一致的数据加载器；评估集用于准确率评估
    train_loader, val_loader, test_loader = get_mnist_dataloaders(cfg, download=True)

    # 构建模型并先做一次虚拟前向，初始化动态 buffer（如 rp_map_idx）
    model = build_model(cfg).to(device)
    model.eval()
    with torch.no_grad():
        dummy = torch.zeros(8, 1, cfg.get('image_size', 28), cfg.get('image_size', 28), device=device)
        _ = model(dummy)
    # 再加载权重
    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f'Checkpoint not found: {args.ckpt}')
    state = torch.load(args.ckpt, map_location=device)
    raw = state.get('model_state_dict', state)
    missing, unexpected = model.load_state_dict(raw, strict=False)
    if missing or unexpected:
        print(f'[warn] load_state_dict mismatches: missing={len(missing)} unexpected={len(unexpected)}')

    # 复校准前评估（若无验证集，则使用测试集）
    eval_loader = val_loader if val_loader is not None else test_loader
    split_name = 'VAL' if val_loader is not None else 'TEST'
    acc_before = evaluate_accuracy(model, eval_loader, device)
    print(f'ACC_BEFORE_BN_CALIB_{split_name}={acc_before:.4f}%')

    # BN 复校准（硬路径）
    bn_recalibrate(model, train_loader, device, max_batches=args.max_batches)

    # 复校准后评估
    acc_after = evaluate_accuracy(model, eval_loader, device)
    print(f'ACC_AFTER_BN_CALIB_{split_name}={acc_after:.4f}%')


if __name__ == '__main__':
    main()


