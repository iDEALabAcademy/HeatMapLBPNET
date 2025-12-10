import os
import torch
from lbpnet.models import build_model
from train_original_model import get_config


def test_dummy_forward_then_load():
    os.environ['MODEL_PRESET'] = 'paper_mnist_rp'
    cfg = get_config()
    model = build_model(cfg)
    x = torch.randn(2, 1, 28, 28)
    # 先 dummy forward 初始化运行时 buffer（如 rp_weights）
    _ = model(x)
    # 模拟不完整 ckpt 加载
    state = model.state_dict()
    model.load_state_dict(state, strict=False)
    with torch.no_grad():
        y = model(x)
    assert torch.isfinite(y).all()



