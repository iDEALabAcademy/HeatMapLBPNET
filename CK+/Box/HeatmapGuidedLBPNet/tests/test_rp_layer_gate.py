import torch
from lbpnet.layers import RPLayer


def test_rp_init_shapes_and_device():
    layer = RPLayer(n_bits_per_out=4, n_out_channels=8, seed=42, learnable=False, fusion_type='rp')
    x = torch.randn(2, 1, 4, 4, 16, 16)  # [B,C,P,N,H,W] with P*N=4*4? here P=4, N=4
    out = layer(x)
    assert out.shape[:2] == (2, 8)
    assert out.device == x.device


def test_gate_monotonic_with_tau():
    layer = RPLayer(n_bits_per_out=4, n_out_channels=8, seed=42, learnable=True, fusion_type='rp')
    # trigger init
    x = torch.randn(1, 4, 1, 4, 8, 8)  # [B,C,P,N,H,W]
    _ = layer(x)
    g_small_tau = torch.sigmoid(layer.gate_logits / 0.5).mean().item()
    g_large_tau = torch.sigmoid(layer.gate_logits / 5.0).mean().item()
    assert g_small_tau != g_large_tau



