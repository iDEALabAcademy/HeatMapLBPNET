import os
import torch
import numpy as np
from torch.utils.data import DataLoader

from train_original_model import get_config
from lbpnet.data import get_mnist_datasets
from lbpnet.models import build_model


def test_val_has_no_random_augment_and_norm_match():
    cfg = get_config()
    train_ds, val_ds, _ = get_mnist_datasets(cfg['data'])
    # sample a few batches to ensure deterministic val transform
    la = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0)
    xb1, yb1 = next(iter(la))
    la = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0)
    xb2, yb2 = next(iter(la))
    # val 数据应稳定可复现（同一次启动下重取首批一致）
    assert torch.allclose(xb1, xb2)
    assert torch.equal(yb1, yb2)
    # 均值方差合理（匹配 Normalize(0.1307,0.3081) 后的分布范围）
    assert torch.isfinite(xb1).all()
    assert xb1.abs().mean().item() < 2.0


def test_class_histogram_balance_within_2pct():
    cfg = get_config()
    train_ds, val_ds, _ = get_mnist_datasets(cfg['data'])
    def hist(ds, num=2000):
        la = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)
        ys = []
        seen = 0
        for _, y in la:
            ys.append(y)
            seen += y.numel()
            if seen >= num:
                break
        y = torch.cat(ys)[:num]
        h = torch.bincount(y, minlength=10).float()
        return h / h.sum().clamp(min=1)
    ht, hv = hist(train_ds), hist(val_ds)
    diff = (ht - hv).abs().max().item()
    assert diff <= 0.02 + 1e-6


def test_state_buffers_saved_and_loaded_equivalent_eval():
    os.environ['MODEL_PRESET'] = 'paper_mnist_rp'
    cfg = get_config()
    model = build_model(cfg)
    model.eval()
    x = torch.randn(2, 1, 28, 28)
    with torch.no_grad():
        y0 = model(x)
    # 保存并加载到新模型
    state = model.state_dict()
    model2 = build_model(cfg)
    # 先 dummy 前向以初始化运行期 buffer（例如 rp_map_idx）
    _ = model2(x)
    model2.load_state_dict(state, strict=False)
    model2.eval()
    with torch.no_grad():
        y1 = model2(x)
    # 允许极小误差
    torch.testing.assert_close(y0, y1, atol=1e-5, rtol=1e-5)


def test_amp_off_for_bn_calibration_eval_path():
    os.environ['MODEL_PRESET'] = 'paper_mnist_rp'
    cfg = get_config()
    model = build_model(cfg)
    model.eval()
    x = torch.randn(4, 1, 28, 28)
    # 明确在 no autocast 下运行一次
    with torch.no_grad():
        y32 = model(x.float())
    # 若系统默认 autocast 下也跑一次，比较数量级接近
    with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
        y_maybe = model(x)
    diff = (y32 - y_maybe.float()).abs().mean().item()
    assert diff < 1e-3
