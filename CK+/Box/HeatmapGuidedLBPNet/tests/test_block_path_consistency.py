import torch
from lbpnet.blocks import MACFreeBlock


def test_block_eval_consistency():
    lbp_cfg = dict(num_patterns=1, num_points=4, window=5, mode='bits', share_across_channels=True)
    rp_cfg = dict(type='rp', n_bits_per_out=4, seed=42, learnable=False)
    block = MACFreeBlock(in_channels=1, out_channels=8, lbp_config=lbp_cfg, rp_config=rp_cfg, downsample=False)
    x = torch.randn(2, 1, 28, 28)
    block.eval()
    with torch.no_grad():
        y1 = block(x)
        y2 = block(x)
    # eval 下应稳定一致
    assert torch.isfinite(y1).all()
    assert torch.allclose(y1, y2)


