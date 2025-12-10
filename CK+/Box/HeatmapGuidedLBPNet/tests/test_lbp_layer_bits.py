import torch
from lbpnet.layers import LBPLayer


def test_lbp_eval_bit_binary_and_deterministic():
    layer = LBPLayer(num_patterns=1, num_points=4, window=5, mode='bits', share_across_channels=True)
    x = torch.randn(2, 1, 28, 28)
    layer.eval()
    with torch.no_grad():
        y1 = layer(x)
        y2 = layer(x)
    # eval 下应确定性一致，且非负（因 pattern_weights 使用 softplus）
    assert torch.isfinite(y1).all()
    assert (y1 >= 0).all()
    assert torch.equal(y1, y2)


