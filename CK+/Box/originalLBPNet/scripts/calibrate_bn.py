#!/usr/bin/env python3
import os
import sys
import torch
from torch.utils.data import DataLoader

# 保证从仓库根目录可导入 lbpnet 包
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from lbpnet.models import build_model
from train_original_model import get_config
from lbpnet.data import get_mnist_datasets


def bn_recalibrate(model, loader, device, max_batches=200):
    model.train()
    # 硬路径前向
    if hasattr(model, 'set_ste'):
        model.set_ste(True, True)
    seen = 0
    with torch.no_grad():
        for data, _ in loader:
            data = data.to(device)
            _ = model(data)
            seen += 1
            if seen >= max_batches:
                break
    model.eval()


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = get_config()
    datasets = get_mnist_datasets(cfg['data'])
    train_dataset = datasets[0]
    loader = DataLoader(train_dataset, batch_size=cfg['training']['batch_size'], shuffle=True, num_workers=0)
    model = build_model(cfg).to(device)

    # 若有最佳权重则加载
    ckpt = os.path.join('./outputs_mnist_original', 'best_model.pth')
    if os.path.exists(ckpt):
        state = torch.load(ckpt, map_location=device)
        raw = state.get('model_state_dict', state)
        try:
            model.load_state_dict(raw, strict=False)
            print(f"已加载: {ckpt}")
        except Exception as e:
            print(f"加载失败（忽略）: {e}")

    bn_recalibrate(model, loader, device, max_batches=200)
    print("BN 重校准完成（硬路径）")


if __name__ == '__main__':
    main()


