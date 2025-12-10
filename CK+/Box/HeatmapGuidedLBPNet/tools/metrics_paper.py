import torch
import torch.nn as nn
from lbpnet.layers import LBPLayer
from lbpnet.layers.rp_paper_layer import RPFusionPaper


def iter_paper_modules(model):
    for m in model.modules():
        if isinstance(m, LBPLayer):
            yield m
        if isinstance(m, RPFusionPaper):
            yield m
        if isinstance(m, nn.Conv2d) and getattr(m, '_is_paper_fusion', False):
            yield m


def get_paper_model_size_bytes(model) -> int:
    """论文口径 Size：仅统计 LBP 采样坐标与 RP 固定映射表/1×1权重的实际序列化字节数。

    - LBP：offsets_raw（或其投影后的离散/浮点坐标）按 float32 计；pattern_weights 与 alpha 等忽略。
    - RP-Paper：rp_map_idx（int32）按实际索引表计；
    - Conv1x1（论文消融用）：权重按 float32 计。
    """
    size = 0
    for m in model.modules():
        if isinstance(m, LBPLayer):
            st = m.state_dict()
            # 仅计 offsets_raw
            if 'offsets_raw' in st:
                t = st['offsets_raw']
                size += int(t.numel() * 4)  # float32
        elif isinstance(m, RPFusionPaper):
            st = m.state_dict()
            if 'rp_map_idx' in st and st['rp_map_idx'].numel() > 0:
                t = st['rp_map_idx']
                # 以 int32 序列化
                size += int(t.numel() * 4)
        elif isinstance(m, nn.Conv2d) and getattr(m, '_is_paper_fusion', False):
            # 1×1 卷积权重
            w = m.weight
            size += int(w.numel() * 4)
    return size


@torch.no_grad()
def estimate_ops_paper(model, input_shape=(1, 1, 28, 28)):
    device = next(model.parameters()).device
    dummy = torch.zeros(*input_shape, device=device)
    add_ops = mul_ops = cmp_ops = 0

    def hook_lbp(m: LBPLayer, inp, out):
        nonlocal cmp_ops, add_ops
        B, P, N, H, W = out.shape
        # 每个 LBP 比特包含一次差分（视作一次加/减法）+ 一次比较
        add_ops += P * N * H * W
        # Comparisons counted in RP hook with C_out

    def hook_rp_paper(m: RPFusionPaper, inp, out):
        nonlocal add_ops, cmp_ops
        x = inp[0]
        if x.dim() == 5:
            B, P, N, H, W = x.shape
        else:
            B, _, H, W = x.shape
        C_out = m.C_out
        k = m.k
        # RP accumulation additions
        add_ops += (k - 1) * C_out * H * W
        # Paper formula: Comp = H × W × C_out × N_p × P
        # LBP comparisons (per output channel): P × N × H × W × C_out
        if x.dim() == 5:
            cmp_ops += C_out * N * P * H * W  # LBP comparisons
        # RP gating comparisons
        cmp_ops += 1 * C_out * H * W

    def hook_conv1x1(m: nn.Conv2d, inp, out):
        nonlocal add_ops, mul_ops
        if not getattr(m, '_is_paper_fusion', False):
            return
        x = inp[0]
        B, C_in, H, W = x.shape
        C_out = m.out_channels
        mul_ops += C_in * C_out * H * W
        add_ops += C_in * C_out * H * W

    hs = []
    for mod in model.modules():
        if isinstance(mod, LBPLayer):
            hs.append(mod.register_forward_hook(hook_lbp))
        if isinstance(mod, RPFusionPaper):
            hs.append(mod.register_forward_hook(hook_rp_paper))
        if isinstance(mod, nn.Conv2d):
            hs.append(mod.register_forward_hook(hook_conv1x1))

    model.eval()
    _ = model(dummy)
    for h in hs:
        h.remove()

    return {
        'adds': add_ops,
        'muls': mul_ops,
        'cmps': cmp_ops,
        'gops_total': (add_ops + mul_ops + cmp_ops) / 1e9,
    }


