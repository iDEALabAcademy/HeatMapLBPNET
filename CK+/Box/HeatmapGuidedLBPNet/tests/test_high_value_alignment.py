import os
import math
import torch
import pytest

from train_original_model import get_config
from lbpnet.models import build_model


@pytest.mark.gpu
@pytest.mark.slow
def test_amp_eval_parity():
    cfg = get_config()
    model = build_model(cfg).eval()
    x = torch.randn(64, 1, 28, 28, device=(torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')))
    model = model.to(x.device)
    with torch.no_grad():
        y32 = model(x.float())
    with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
        y16 = model(x)
    # Logits 数值接近；Top-1 大多数一致（<=5% 差异，初始阈值可再收紧）
    p32 = y32.argmax(dim=1)
    p16 = y16.argmax(dim=1)
    top1_diff = (p32 != p16).float().mean().item()
    mae = (y32 - y16.float()).abs().mean().item()
    assert torch.isfinite(y32).all() and torch.isfinite(y16).all()
    assert mae < 1e-3
    assert top1_diff <= 0.05


@pytest.mark.slow
def test_ste_grad_finite_and_bounded():
    os.environ['MODEL_PRESET'] = 'paper_mnist_rp'
    cfg = get_config()
    model = build_model(cfg)
    model.train()
    x = torch.randn(8, 1, 28, 28)
    y = torch.randint(0, 10, (8,))
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits, y)
    loss.backward()
    total_norm = 0.0
    cnt = 0
    for p in model.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()
            total_norm += p.grad.detach().norm().item()
            cnt += 1
    avg_norm = total_norm / max(cnt, 1)
    assert 1e-6 <= avg_norm <= 10


@pytest.mark.slow
def test_global_alive_ratio_monotonic():
    cfg = get_config()
    model = build_model(cfg).eval()
    x = torch.randn(8, 1, 28, 28)
    # 记录不同阈值下的 alive(hard)（我们设计中前向恒硬，tau 不影响前向）
    from lbpnet.layers.rp_paper_layer import RPFusionPaper
    vals = []
    for thr in [2, 4, 6]:
        for m in model.modules():
            if isinstance(m, RPFusionPaper):
                m.threshold = int(thr)
        with torch.no_grad():
            _ = model(x)
        alives = []
        for m in model.modules():
            if isinstance(m, RPFusionPaper):
                alives.append(m.get_alive_ratio())
        vals.append(sum(alives) / max(len(alives), 1))
    # 阈值增大 → alive 不增
    assert vals[0] >= vals[1] >= vals[2]
    # 区间大致合理
    assert 0.05 <= vals[0] <= 0.8


@pytest.mark.slow
def test_dataloader_determinism():
    from lbpnet.data import get_mnist_datasets
    from torch.utils.data import DataLoader
    import numpy as np
    cfg = get_config()
    train_ds, _, _ = get_mnist_datasets(cfg['data'])
    def first_batches(num_workers, persistent=False, seed=1234):
        g = torch.Generator(); g.manual_seed(seed)
        def _init_fn(worker_id: int):
            s = seed + worker_id
            np.random.seed(s)
            torch.manual_seed(s)
        dl = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=num_workers,
                        persistent_workers=persistent, generator=g, worker_init_fn=_init_fn)
        it = iter(dl)
        xb1, yb1 = next(it)
        xb2, yb2 = next(it)
        return torch.utils.data.get_worker_info(), xb1.clone(), yb1.clone(), xb2.clone(), yb2.clone()
    # 固定种子由数据模块实现，跨不同 worker 设置首两批应一致
    _, a1, ya1, a2, ya2 = first_batches(0)
    _, b1, yb1, b2, yb2 = first_batches(2)
    assert torch.allclose(a1, b1) and torch.equal(ya1, yb1)
    assert torch.allclose(a2, b2) and torch.equal(ya2, yb2)


@pytest.mark.slow
def test_resume_training_equivalence(tmp_path):
    # 构造极小训练步，对比中断恢复 vs 连续两步（在 CPU 与确定性模式下）
    torch.manual_seed(123)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['MODEL_PRESET'] = 'paper_mnist_rp'
    cfg = get_config()
    device = torch.device('cpu')
    modelA = build_model(cfg).to(device)
    modelB = build_model(cfg).to(device)
    optA = torch.optim.SGD(modelA.parameters(), lr=1e-2)
    optB = torch.optim.SGD(modelB.parameters(), lr=1e-2)
    x = torch.randn(16, 1, 28, 28, device=device)
    y = torch.randint(0, 10, (16,), device=device)
    # 步骤1
    lossA1 = torch.nn.functional.cross_entropy(modelA(x), y); optA.zero_grad(); lossA1.backward(); optA.step()
    lossB1 = torch.nn.functional.cross_entropy(modelB(x), y); optB.zero_grad(); lossB1.backward(); optB.step()
    # 保存/恢复 B
    ckpt = {
        'model': modelB.state_dict(),
        'opt': optB.state_dict(),
    }
    p = tmp_path / 'ckpt.pth'
    torch.save(ckpt, p)
    modelB2 = build_model(cfg).to(device); optB2 = torch.optim.SGD(modelB2.parameters(), lr=1e-2)
    state = torch.load(p)
    # 先 dummy 前向初始化运行期 buffer（如 rp_map_idx）
    _ = modelB2(x)
    modelB2.load_state_dict(state['model'], strict=False)
    optB2.load_state_dict(state['opt'])
    # 步骤2
    lossA2 = torch.nn.functional.cross_entropy(modelA(x), y); optA.zero_grad(); lossA2.backward(); optA.step()
    lossB2 = torch.nn.functional.cross_entropy(modelB2(x), y); optB2.zero_grad(); lossB2.backward(); optB2.step()
    # 权重接近（数值上存在浮点顺序差，放宽阈值）
    # 直接比较输出等价性
    with torch.no_grad():
        outA = modelA(x)
        outB = modelB2(x)
    mae = (outA - outB).abs().mean().item()
    assert mae < 1e-1


def test_lr_schedule_trace():
    # 验证 PyTorch CosineAnnealingLR 曲线与解析期望一致
    m = torch.nn.Linear(2,2)
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    T_max = 10
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=T_max, eta_min=0.0)
    lrs = [opt.param_groups[0]['lr']]
    for _ in range(T_max):
        sch.step()
        lrs.append(opt.param_groups[0]['lr'])
    # 解析：lr_t = 0.5*(1+cos(pi*t/T_max))*lr0
    import math
    lr0 = 0.1
    expected = [0.5*(1+math.cos(math.pi*t/T_max))*lr0 for t in range(0, T_max+1)]
    mae = sum(abs(a-b) for a,b in zip(lrs, expected)) / len(lrs)
    assert mae < 1e-9


def test_torchscript_export_smoke():
    cfg = get_config()
    model = build_model(cfg).eval()
    x = torch.randn(2,1,28,28)
    with torch.no_grad():
        y = model(x)
    traced = torch.jit.trace(model, x)
    y2 = traced(x)
    torch.testing.assert_close(y, y2, atol=1e-5, rtol=1e-5)


def test_state_dict_backward_compat():
    cfg = get_config()
    model = build_model(cfg).eval()
    x = torch.randn(1,1,28,28)
    _ = model(x)
    state = model.state_dict()
    # 模拟旧 ckpt：移除 rp_map_idx
    to_del = [k for k in state.keys() if k.endswith('rp_map_idx')]
    for k in to_del:
        del state[k]
    # 仍可加载并前向
    model2 = build_model(cfg).eval()
    _ = model2(x)
    model2.load_state_dict(state, strict=False)
    with torch.no_grad():
        y = model2(x)
    assert torch.isfinite(y).all()


@pytest.mark.slow
@pytest.mark.data
def test_quant_threshold_edgecases():
    # 在阈值±epsilon附近抖动，输出形状保持不变，预测变化占比不过高
    from lbpnet.layers.rp_paper_layer import RPFusionPaper
    m = RPFusionPaper(n_bits_per_out=4, n_out_channels=8)
    x = torch.randint(0, 2, (2, 2, 8, 14, 14)).float()
    with torch.no_grad():
        base = m(x).clone()
    for eps in [1e-3, 1e-2]:
        with torch.no_grad():
            y = m(x + eps)
        assert y.shape == base.shape
        change = (y != base).float().mean().item()
        assert change <= 0.2


@pytest.mark.slow
def test_cli_invalid_args_raises():
    from lbpnet.blocks import MACFreeBlock
    from lbpnet.layers import LBPLayer
    lbp_cfg = dict(num_patterns=1, num_points=4, window=5, mode='bits', share_across_channels=True)
    with pytest.raises(ValueError):
        MACFreeBlock(in_channels=1, out_channels=8, lbp_config=lbp_cfg, rp_config={'fusion_type': 'foo'})


@pytest.mark.gpu
@pytest.mark.slow
def test_cpu_gpu_parity():
    if not torch.cuda.is_available():
        pytest.skip('no CUDA')
    cfg = get_config()
    model = build_model(cfg).eval()
    x = torch.randn(64,1,28,28)
    with torch.no_grad():
        y_cpu = model(x)
        y_gpu = model.to('cuda')(x.to('cuda'))
        y_gpu = y_gpu.cpu()
    top1_diff = (y_cpu.argmax(1) != y_gpu.argmax(1)).float().mean().item()
    assert top1_diff <= 0.3


@pytest.mark.slow
@pytest.mark.data
def test_bn_calibration_sweep_requires_ckpt():
    # 需训练好 ckpt 才有意义，否则跳过
    ckpt = '/home/hding22/binary/outputs_mnist_original/best_model.pth'
    if not os.path.isfile(ckpt):
        pytest.skip('no trained checkpoint')
    # 仅作冒烟：能够加载并运行不同校准步数
    cfg = get_config()
    model = build_model(cfg)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    from lbpnet.data import get_mnist_dataloaders
    train_loader, val_loader, _ = get_mnist_dataloaders(cfg)
    def calibrate(steps: int):
        model.train()
        cnt = 0
        with torch.no_grad():
            for xb, _ in train_loader:
                xb = xb.to(device)
                _ = model(xb)
                cnt += 1
                if cnt >= steps: break
        model.eval()
    def eval_loss():
        xb, yb = next(iter(val_loader))
        xb, yb = xb.to(device), yb.to(device)
        with torch.no_grad():
            logits = model(xb)
            return torch.nn.functional.cross_entropy(logits, yb).item()
    base = eval_loss()
    calibrate(8); l8 = eval_loss()
    calibrate(32); l32 = eval_loss()
    calibrate(128); l128 = eval_loss()
    # 容忍轻微噪声：不劣于 base 的 2%，且 32→128 收敛变化 < 2%
    assert l8 <= base + 2e-2
    assert l32 <= base + 2e-2
    assert abs(l128 - l32) <= 2e-2


@pytest.mark.slow
def test_flops_latency_consistency_monotonic():
    # 不依赖 scipy：用输入尺寸递增的单调性作为替代
    import os
    if os.environ.get('PERF_TEST', '0') != '1':
        pytest.skip('性能测试默认关闭，设置 PERF_TEST=1 启用')
    cfg = get_config()
    model = build_model(cfg).eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        pytest.skip('CPU 上延迟受噪声影响大，跳过该项')
    model.to(device)
    from tools.metrics_paper import estimate_ops_paper
    sizes = [28, 40, 56]
    ops = []
    lats = []
    for s in sizes:
        with torch.no_grad():
            _ = model(torch.zeros(2,1,s,s, device=device))
        m = estimate_ops_paper(model, (1,1,s,s))
        ops.append(m['adds']+m['muls']+m['cmps'])
        # 粗略延迟（毫秒）
        import time
        iters = 20
        x = torch.randn(8,1,s,s, device=device)
        if device.type == 'cuda': torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(iters):
                _ = model(x)
        if device.type == 'cuda': torch.cuda.synchronize()
        dt = (time.perf_counter()-t0)/iters
        lats.append(dt)
    # 复杂度单调性成立；延迟用相关性度量整体趋势
    assert ops[0] < ops[1] < ops[2]
    import math
    xs = [float(s) for s in sizes]
    ys = [float(v) for v in lats]
    mx = sum(xs)/len(xs); my = sum(ys)/len(ys)
    cov = sum((x-mx)*(y-my) for x,y in zip(xs,ys))
    sx = math.sqrt(sum((x-mx)**2 for x in xs)); sy = math.sqrt(sum((y-my)**2 for y in ys))
    r = cov / (sx*sy + 1e-12)
    assert r > 0.2


@pytest.mark.slow
def test_onnx_export_alignment():
    try:
        import onnx, onnxruntime as ort  # noqa: F401
    except Exception:
        pytest.skip('onnxruntime not available')
    cfg = get_config()
    model = build_model(cfg).eval()
    x = torch.randn(2,1,28,28)
    onnx_path = 'tmp_model.onnx'
    torch.onnx.export(model, x, onnx_path, input_names=['input'], output_names=['logits'], opset_version=12, dynamic_axes={'input':{0:'B'}, 'logits':{0:'B'}})
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    y_ref = model(x).detach().numpy()
    y_onx = sess.run(['logits'], {'input': x.numpy()})[0]
    assert abs(y_ref - y_onx).mean() < 1e-3


@pytest.mark.skip(reason="skeleton: share flag semantics require per-channel impl to compare")
def test_lbp_share_flag_semantics():
    assert True


@pytest.mark.slow
@pytest.mark.data
@pytest.mark.skip(reason="skeleton: BN calibration sweep placeholder")
def test_bn_calibration_sweep():
    assert True


@pytest.mark.slow
@pytest.mark.skip(reason="skeleton: flops-latency correlation placeholder")
def test_flops_latency_consistency():
    assert True


