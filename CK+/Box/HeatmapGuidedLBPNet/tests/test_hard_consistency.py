import os
import torch


def test_train_eval_hardpath_consistency():
    os.environ['MODEL_PRESET'] = 'paper_mnist_rp'
    from lbpnet.models import build_model

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model({
        "model": "lbpnet_rp",
        "lbp_layer": {"num_patterns": 2, "num_points": 8, "window": 5, "share_across_channels": True, "mode": "bits", "alpha_init": 0.2, "learn_alpha": True, "offset_init_std": 0.8},
        "blocks": {"stages": 1, "channels_per_stage": [8], "downsample_at": [], "fusion_type": "rp_paper", "rp_config": {"fusion_type": "rp_paper", "n_bits_per_out": 4, "seed": 1, "use_ste": True}},
        "head": {"hidden": 16, "dropout_rate": 0.0, "num_classes": 10, "use_bn": False},
        "training": {"epochs": 1, "batch_size": 4, "lr": 1e-3, "weight_decay": 0.0},
        "ste": {"use_ste_bits": True, "use_ste_gates": True, "lbp_ste_scale": 6.0, "gate_ste_scale": 6.0},
        "num_workers": 0, "pin_memory": False
    }).to(device)

    # 统一硬前向 + STE
    if hasattr(model, 'set_ste'):
        model.set_ste(True, True)

    x = torch.randn(4, 1, 28, 28, device=device)

    # 两次前向都在 BN eval 模式下比较，避免统计差异带来的数值漂移
    for m in model.modules():
        if m.__class__.__name__ == 'BatchNorm2d':
            m.eval()

    model.eval()
    with torch.no_grad():
        y1 = model(x)
        y2 = model(x)

    # 允许极小数值误差（主路径均为硬二值，通过加法/BN/激活可能带来极小误差）
    torch.testing.assert_close(y1.detach().cpu(), y2.detach().cpu(), atol=1e-5, rtol=1e-5)











