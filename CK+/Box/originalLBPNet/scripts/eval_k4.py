#!/usr/bin/env python3
import os
import sys
import time
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from train_original_model import get_config
from lbpnet.models import build_model
from SVHN.Box.binary_Ding.lbpnet.data.svhn_dataset import get_mnist_dataloaders
from tools.metrics_paper import estimate_ops_paper


def eval_acc(model: torch.nn.Module, loader, device: torch.device) -> float:
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            out = model(x)
            pred = out.argmax(1)
            total += int(y.numel())
            correct += int((pred == y).sum())
    return 100.0 * correct / max(1, total)


def main():
    device = torch.device('cpu')
    os.environ['MODEL_PRESET'] = 'paper_mnist_rp_full'  # k=4 预设
    cfg = get_config()
    print('[INFO] preset=paper_mnist_rp_full (k=4), device=', device)

    train_loader, val_loader, test_loader = get_mnist_dataloaders(cfg, download=False)

    model = build_model(cfg).to(device)
    model.eval()
    H = W = int(cfg.get('image_size', 28))
    with torch.no_grad():
        _ = model(torch.zeros(8, 1, H, W, device=device))

    # 加载 k=4 的 best 权重（若存在）
    ckpt_path = os.path.abspath('./training_results_archive/outputs_paper_rp_full60k_v1/best_model.pth')
    try:
        ckpt = torch.load(ckpt_path, map_location=device)
        state = ckpt.get('model_state_dict', ckpt)
        incompatible = model.load_state_dict(state, strict=False)
        print('[INFO] loaded checkpoint:', ckpt_path,
              'missing=', len(getattr(incompatible, 'missing_keys', [])),
              'unexpected=', len(getattr(incompatible, 'unexpected_keys', [])))
    except Exception as e:
        print('[WARN] no checkpoint or load failed:', e)

    ops = estimate_ops_paper(model, (1, 1, H, W))
    print(f"[PAPER] gops={ops['gops_total']:.6f}, ops(cmp/add/mul)={ops['cmps']}/{ops['adds']}/{ops['muls']}")

    test_acc = eval_acc(model, test_loader, device)
    print(f"TEST_ACC {test_acc:.4f}%")
    print('[DONE] eval finished at', time.strftime('%Y-%m-%d %H:%M:%S'))


if __name__ == '__main__':
    main()






