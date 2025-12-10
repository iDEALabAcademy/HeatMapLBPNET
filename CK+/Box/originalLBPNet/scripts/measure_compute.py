#!/usr/bin/env python3
import os
import sys
import json
import time
import argparse
import torch
import torch.nn.functional as F

# 保证从仓库根目录可导入 lbpnet 包
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from lbpnet.models import build_model
from train_original_model import get_config


def measure_latency(model, device, image_size=28, iters=50, warmup=20, batch_size=128):
    x = torch.randn(batch_size, 1, image_size, image_size, device=device)
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
        torch.cuda.synchronize() if device.type=='cuda' else None
        t0 = time.time()
        for _ in range(iters):
            _ = model(x)
        torch.cuda.synchronize() if device.type=='cuda' else None
        t1 = time.time()
    avg_ms = (t1 - t0) * 1000.0 / iters
    ips = batch_size * 1000.0 / avg_ms
    return avg_ms, ips


def estimate_params_bytes(model) -> int:
    return sum(p.numel() for p in model.parameters()) * 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--preset', type=str, default=os.environ.get('MODEL_PRESET','paper_mnist_rp'))
    ap.add_argument('--output', type=str, default='metrics.json')
    args = ap.parse_args()

    os.environ['MODEL_PRESET'] = args.preset
    cfg = get_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(cfg).to(device)

    # 简易准确率占位（如已训练可读最佳权重）
    acc = None
    ckpt = os.path.join('./outputs_mnist_original', 'best_model.pth')
    if os.path.exists(ckpt):
        try:
            state = torch.load(ckpt, map_location=device)
            raw = state.get('model_state_dict', state)
            model.load_state_dict(raw, strict=False)
            acc = float(state.get('best_val_acc', 0.0))
        except Exception:
            pass

    # 延迟与吞吐
    lat_ms, ips = measure_latency(model, device, image_size=cfg.get('image_size',28))

    # 估算参数规模（fp32）
    size_mb_est = estimate_params_bytes(model) / (1024.0**2)

    metrics = {
        'dataset': 'MNIST',
        'preset': args.preset,
        'accuracy_%': acc,
        'error_rate_%': (100.0 - acc) if acc is not None else None,
        'num_params': sum(p.numel() for p in model.parameters()),
        'model_size_mb_fp32_est': size_mb_est,
        'checkpoint_size_mb': (os.path.getsize(ckpt)/(1024.0**2)) if os.path.exists(ckpt) else None,
        'latency_ms_gpu' if device.type=='cuda' else 'latency_ms_cpu': lat_ms,
        'throughput_ips_gpu' if device.type=='cuda' else 'throughput_ips_cpu': ips,
    }

    with open(args.output,'w') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()


