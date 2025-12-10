#!/usr/bin/env python3
import os
import sys
from collections import defaultdict

import torch
import torch.nn as nn

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from train_original_model import get_config
from lbpnet.models import build_model
from lbpnet.layers.lbp_layer import LBPLayer
from lbpnet.layers.rp_paper_layer import RPFusionPaper


def shape_of(x):
    if isinstance(x, torch.Tensor):
        return list(x.shape)
    if isinstance(x, (list, tuple)) and x and isinstance(x[0], torch.Tensor):
        return [list(t.shape) for t in x]
    return str(type(x))


def main():
    preset = os.environ.get("MODEL_PRESET", "paper_mnist_rp")
    cfg = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device).eval()

    H = W = int(cfg.get("image_size", 28))
    dummy = torch.randn(2, 1, H, W, device=device)

    records = []

    def hook_fn(name):
        def _hook(m, inp, out):
            x = inp[0] if isinstance(inp, (list, tuple)) and len(inp) > 0 else inp
            entry = {
                "module": name,
                "type": m.__class__.__name__,
                "in": shape_of(x),
                "out": shape_of(out),
                "dtype": str((out if isinstance(out, torch.Tensor) else x).dtype if isinstance(out, torch.Tensor) or isinstance(x, torch.Tensor) else "n/a"),
                "device": str((out if isinstance(out, torch.Tensor) else x).device if isinstance(out, torch.Tensor) or isinstance(x, torch.Tensor) else "n/a"),
            }
            # Special notes for expected forms
            if isinstance(m, LBPLayer):
                entry["expect"] = "LBP bits [B,P,N,H,W]"
            elif isinstance(m, RPFusionPaper):
                entry["expect"] = "RP out [B,C,H,W]"
            elif isinstance(m, nn.AdaptiveAvgPool2d):
                entry["expect"] = "GAP -> [B,C,1,1]"
            elif isinstance(m, nn.Linear):
                entry["expect"] = f"Linear out [B,{m.out_features}]"
            elif isinstance(m, nn.AvgPool2d):
                entry["expect"] = "Downsample H,W/2"
            records.append(entry)
        return _hook

    # Register hooks for key modules
    for name, m in model.named_modules():
        if isinstance(m, (LBPLayer, RPFusionPaper, nn.Conv2d, nn.BatchNorm2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d, nn.Linear)):
            m.register_forward_hook(hook_fn(name))

    with torch.no_grad():
        _ = model(dummy)

    # Group by stage prefix for readability
    def stage_key(n):
        if n.startswith("stages."):
            return n.split(".")[1]
        return "head"

    grouped = defaultdict(list)
    for r in records:
        grouped[stage_key(r["module"])].append(r)

    print(f"[CHECK] preset={preset}, device={device}, image={H}x{W}")
    # Print in order
    for k in sorted(grouped.keys(), key=lambda x: (x!='head', x)):
        print(f"\n== Stage {k} ==")
        for r in grouped[k]:
            print(f"- {r['module']:<45} | {r['type']:<18} | in={r['in']} -> out={r['out']} | {r.get('expect','')}")

    # Lightweight assertions
    issues = []
    # LBP -> RPFusionPaper channel compatibility
    for i in range(3):
        lbp_key = f"stages.{i}.lbp_layer"
        rp_key = f"stages.{i}.fuse"
        lbp_out = next((r for r in records if r["module"] == lbp_key), None)
        rp_in = next((r for r in records if r["module"] == rp_key), None)
        if lbp_out and rp_in:
            if not (isinstance(lbp_out["out"], list) and len(lbp_out["out"])==5):
                issues.append(f"LBP output shape unexpected at {lbp_key}: {lbp_out['out']}")
            if not (isinstance(rp_in["in"], list) and (len(rp_in["in"]) in (4,5))):
                issues.append(f"RP input shape unexpected at {rp_key}: {rp_in['in']}")

    # Report
    if issues:
        print("\n[CHECK][ISSUES]")
        for it in issues:
            print("- ", it)
        sys.exit(2)
    else:
        print("\n[CHECK] alignment looks consistent across key modules.")


if __name__ == "__main__":
    main()






