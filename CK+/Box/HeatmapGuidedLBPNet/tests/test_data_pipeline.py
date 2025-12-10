import os
import torch
from torch.utils.data import DataLoader

from train_original_model import get_config
from lbpnet.data import get_mnist_datasets


def test_mnist_shapes_and_ranges():
    cfg = get_config()
    datasets = get_mnist_datasets(cfg['data'])
    train_dataset = datasets[0]
    loader = DataLoader(train_dataset, batch_size=8, shuffle=False, num_workers=0)
    x, y = next(iter(loader))
    assert x.dim() == 4 and x.shape[1:] == (1, 28, 28)
    assert y.dim() == 1 and x.size(0) == y.size(0)
    assert x.dtype in (torch.float32, torch.float64)
    # 容忍标准化到任意均值方差；但数值应在合理范围
    assert torch.isfinite(x).all()
    assert x.abs().mean().item() < 2.0


def test_seed_reproducibility():
    os.environ['MODEL_PRESET'] = 'paper_mnist_rp'
    cfg = get_config()
    datasets_a = get_mnist_datasets(cfg['data'])
    datasets_b = get_mnist_datasets(cfg['data'])
    la = DataLoader(datasets_a[0], batch_size=8, shuffle=False, num_workers=0)
    lb = DataLoader(datasets_b[0], batch_size=8, shuffle=False, num_workers=0)
    xa, ya = next(iter(la))
    xb, yb = next(iter(lb))
    # 同一随机种子设定下，首批应一致
    assert torch.allclose(xa, xb)
    assert torch.equal(ya, yb)



