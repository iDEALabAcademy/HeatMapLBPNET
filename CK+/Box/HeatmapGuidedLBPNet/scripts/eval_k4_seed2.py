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
    # 使用 CPU 以避免占用训练中的 GPU
    device = torch.device('cpu')

    # 选择 k=4 预设
    os.environ['MODEL_PRESET'] = 'paper_mnist_rp_full'
    cfg = get_config()

    print('[INFO] preset=paper_mnist_rp_full (k=4), device=', device)
    train_loader, val_loader, test_loader = get_mnist_dataloaders(cfg, download=False)

    # 构建基线模型并做一次 dummy forward
    model = build_model(cfg).to(device)
    model.eval()
    H = W = int(cfg.get('image_size', 28))
    with torch.no_grad():
        _ = model(torch.zeros(8, 1, H, W, device=device))

    # 加载已有权重（若存在）
    ckpt_path = os.path.abspath('./training_results_archive/outputs_paper_rp_full60k_v1/best_model.pth')
    state = None
    try:
        ckpt = torch.load(ckpt_path, map_location=device)
        state = ckpt.get('model_state_dict', ckpt)
        incompatible = model.load_state_dict(state, strict=False)
        print('[INFO] loaded k=4 checkpoint:', ckpt_path,
              'missing=', len(getattr(incompatible, 'missing_keys', [])),
              'unexpected=', len(getattr(incompatible, 'unexpected_keys', [])))
    except Exception as e:
        print('[WARN] no checkpoint or load failed:', e)

    # 基线精度与单次 GOPs
    base_acc = eval_acc(model, test_loader, device)
    ops = estimate_ops_paper(model, (1, 1, H, W))
    print(f"BASE_ACC {base_acc:.4f}% GOPs(single) {ops['gops_total']:.6f}")

    # 多 seed=2 集成评估
    seeds = [cfg['blocks']['rp_config'].get('seed', 42), 43]
    models = []
    for sd in seeds:
        m = build_model(cfg).to(device)
        if state is not None:
            try:
                m.load_state_dict(state, strict=False)
            except Exception:
                pass
        m.eval()
        # 以新 seed 重新构建 rp 映射
        for s in m.stages:
            if hasattr(s, 'fuse') and hasattr(s.fuse, '_init_map'):
                s.fuse.seed = sd
                s.fuse._initialized = False
        with torch.no_grad():
            _ = m(torch.zeros(8, 1, H, W, device=device))
        models.append(m)

    @torch.no_grad()
    def eval_acc_ensemble(models, loader) -> float:
        for m in models:
            m.eval()
        total = 0
        correct = 0
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits_sum = None
            for m in models:
                out = m(x)
                logits_sum = out if logits_sum is None else (logits_sum + out)
            pred = logits_sum.argmax(1)
            total += int(y.numel())
            correct += int((pred == y).sum())
        return 100.0 * correct / max(1, total)

    ens_acc = eval_acc_ensemble(models, test_loader)
    print(f"ENSEMBLE2_ACC {ens_acc:.4f}% GOPs(equiv) {2*ops['gops_total']:.6f}")
    print('[DONE] eval finished at', time.strftime('%Y-%m-%d %H:%M:%S'))


if __name__ == '__main__':
    main()






