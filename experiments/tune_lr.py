"""
tune_lr.py — Learning rate sweep for the TinyLSTM online predictor.

Usage:
    python tune_lr.py --file path/to/enwik8 --steps 50000
    python tune_lr.py --synthetic --steps 30000

Sweeps learning rates in [0.001, 0.002, 0.005, 0.01, 0.02, 0.05]
and plots bits/byte vs learning rate.
"""

import argparse
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

from model_arch import TinyLSTM, synthetic_text


LR_CANDIDATES = [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05]


def encode_bits_per_byte(data: bytes, hidden: int, lr: float,
                          steps: int, device: torch.device) -> float:
    model = TinyLSTM(hidden=hidden, embed_dim=hidden).to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    model.train()
    model.reset_state(device=device)

    n = min(len(data), steps + 1)
    total_bits = 0.0
    prev_byte = torch.tensor([0], dtype=torch.long, device=device)

    for i in range(n - 1):
        actual_byte = data[i + 1]
        actual_t    = torch.tensor([actual_byte], dtype=torch.long, device=device)

        probs = model(prev_byte)
        p = max(probs[actual_byte].item(), 1e-30)
        total_bits += -math.log2(p)

        loss = -torch.log(probs[actual_byte] + 1e-30)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.h = model.h.detach()
        model.c = model.c.detach()
        prev_byte = actual_t

    return total_bits / max(n - 1, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file",      type=str,   default=None)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--hidden",    type=int,   default=64)
    parser.add_argument("--steps",     type=int,   default=50_000)
    parser.add_argument("--cpu",       action="store_true")
    args = parser.parse_args()

    device = torch.device("cpu") if args.cpu or not torch.cuda.is_available() else torch.device("cuda")

    if args.file:
        with open(args.file, "rb") as f:
            data = f.read(args.steps + 1)
        print(f"Loaded {len(data):,} bytes from {args.file}")
    else:
        data = synthetic_text(args.steps + 1)
        print(f"Using synthetic text ({len(data):,} bytes)")

    print(f"\nSweeping learning rates (hidden={args.hidden}, steps={args.steps:,})...\n")

    results = {}
    for lr in LR_CANDIDATES:
        t0 = time.perf_counter()
        bpb = encode_bits_per_byte(data, args.hidden, lr, args.steps, device)
        elapsed = time.perf_counter() - t0
        print(f"  lr={lr:.4f}  →  {bpb:.4f} bits/byte  ({elapsed:.1f}s)")
        results[lr] = bpb

    best_lr = min(results, key=results.get)
    print(f"\nBest LR: {best_lr}  ({results[best_lr]:.4f} bits/byte)")

    if HAS_MPL:
        lrs  = list(results.keys())
        bpbs = list(results.values())
        plt.figure(figsize=(8, 5))
        plt.semilogx(lrs, bpbs, marker='o')
        plt.xlabel("Learning rate")
        plt.ylabel("bits/byte")
        plt.title(f"LR sweep (hidden={args.hidden}, steps={args.steps:,})")
        plt.grid(True)
        plt.tight_layout()
        out = "lr_sweep.png"
        plt.savefig(out)
        print(f"Plot saved to {out}")


if __name__ == "__main__":
    main()
