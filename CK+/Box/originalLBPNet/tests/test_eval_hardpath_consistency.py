import torch
from lbpnet.models import build_model
from train_original_model import get_config


def test_eval_hardpath_close_to_train():
    cfg = get_config()
    model = build_model(cfg)
    x = torch.randn(8, 1, 28, 28)
    model.train()
    if hasattr(model, 'set_ste'):
        model.set_ste(True, True)
    y_tr = model(x)
    model.eval()
    with torch.no_grad():
        y_ev = model(x)
    # logits 分布差异不应离谱（宽松阈值）
    diff = (y_tr.mean() - y_ev.mean()).abs().item()
    assert diff < 1.0



