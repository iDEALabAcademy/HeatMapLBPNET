#!/usr/bin/env python3
"""
CLI to measure model size (bytes) with mixed bitwidth policy.

Only counts learnable parameters; excludes buffers (e.g., BN running stats).
Outputs CSV and JSON reports.
"""

import os
import sys
import argparse
import json
import torch

# Ensure local imports work when executed directly
REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from utils.model_size import load_policy, measure_model_size, save_reports, to_human_readable
from lbpnet.models import build_model


from typing import Optional


def load_model_from_ckpt(ckpt_path: Optional[str]) -> torch.nn.Module:
    # Build from default preset in training script to keep structure
    from train_original_model import get_config
    config = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config).to(device)
    if ckpt_path and os.path.isfile(ckpt_path):
        try:
            # dummy forward to init dynamic buffers (e.g., RP maps)
            H = W = int(config.get('image_size', 28))
            with torch.no_grad():
                _ = model(torch.zeros(2, 1, H, W, device=device))
            ckpt = torch.load(ckpt_path, map_location=device)
            state = ckpt.get('model_state_dict', ckpt)
            model.load_state_dict(state, strict=False)
        except Exception as e:
            print(f"[WARN] Failed to load ckpt: {ckpt_path}, err={e}")
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser(description="Measure model size with bitwidth policy")
    ap.add_argument("--ckpt", type=str, default="", help="Path to checkpoint .pth")
    ap.add_argument("--out", type=str, default="./outputs_mnist_original", help="Output directory")
    ap.add_argument("--policy", type=str, default="", help="YAML policy path")
    ap.add_argument("--strict", action="store_true", help="Error if a param has no bitwidth match")
    ap.add_argument("--dry-run", action="store_true", help="Only show matched bitwidth without bytes accumulation")
    args = ap.parse_args()

    policy = load_policy(args.policy)
    model = load_model_from_ckpt(args.ckpt if args.ckpt else None)

    result = measure_model_size(model, policy=policy, strict=bool(args.strict), dry_run=bool(args.dry_run))
    w_bytes = result["weight_only_total_bytes"]
    f_bytes = result["full_params_total_bytes"]
    w_kb, w_mb = to_human_readable(w_bytes)
    f_kb, f_mb = to_human_readable(f_bytes)

    print(f"[MODEL_SIZE] weight_only={w_bytes} Bytes ({w_kb:.2f} KB, {w_mb:.3f} MB), "
          f"full_params={f_bytes} Bytes ({f_kb:.2f} KB, {f_mb:.3f} MB)")

    csv_path, json_path = save_reports(result, args.out)
    print(f"Saved reports: CSV={csv_path}, JSON={json_path}")

    # Bitwidth histogram
    print("Bitwidth histogram:")
    by_bit = result.get("by_bitwidth", {})
    for bit in sorted(by_bit.keys()):
        info = by_bit[bit]
        kb, mb = to_human_readable(info["bytes"])
        print(f"  {bit:>2} bit: params={info['params']:,}, bytes={info['bytes']:,} ({kb:.2f} KB, {mb:.3f} MB)")


if __name__ == "__main__":
    main()










