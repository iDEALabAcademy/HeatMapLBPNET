#!/usr/bin/env python3
import os
import sys
import time
import json
import argparse
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from train_original_model import get_config
from lbpnet.models import build_model
from tools.metrics_paper import estimate_ops_paper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preset', default='paper_mnist_rp_full', type=str)
    parser.add_argument('--seed', default=2, type=int)
    parser.add_argument('--n_bits', default=None, type=int, help='Override rp_config.n_bits_per_out')
    parser.add_argument('--device', default='cpu', type=str)
    parser.add_argument('--log_json', default=None, type=str)
    args = parser.parse_args()

    device = torch.device(args.device)
    os.environ['MODEL_PRESET'] = args.preset
    cfg = get_config()

    # Override seed and n_bits_per_out deterministically
    if 'blocks' in cfg and 'rp_config' in cfg['blocks']:
        cfg['blocks']['rp_config']['seed'] = int(args.seed)
        if args.n_bits is not None:
            cfg['blocks']['rp_config']['n_bits_per_out'] = int(args.n_bits)

    # Build model and do a dummy forward to initialize shapes/buffers
    model = build_model(cfg).to(device)
    model.eval()
    H = W = int(cfg.get('image_size', 28))
    with torch.no_grad():
        _ = model(torch.zeros(8, 1, H, W, device=device))

    # Measure GOPs with paper-aligned estimator
    ops = estimate_ops_paper(model, (1, 1, H, W))
    gops = float(ops['gops_total'])
    print(f"[PAPER] preset={args.preset} seed={args.seed} n_bits={cfg['blocks']['rp_config'].get('n_bits_per_out')} gops={gops:.6f} ops(cmp/add/mul)={ops['cmps']}/{ops['adds']}/{ops['muls']}")

    if args.log_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.log_json)), exist_ok=True)
        with open(args.log_json, 'w') as f:
            json.dump({
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'preset': args.preset,
                'seed': int(args.seed),
                'n_bits': int(cfg['blocks']['rp_config'].get('n_bits_per_out')),
                'gops': gops,
                'ops_breakdown': {
                    'cmps': int(ops['cmps']),
                    'adds': int(ops['adds']),
                    'muls': int(ops['muls'])
                }
            }, f, indent=2)
        print('[SAVED]', args.log_json)


if __name__ == '__main__':
    main()





