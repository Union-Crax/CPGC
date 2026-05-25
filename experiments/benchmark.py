"""
benchmark.py — Compare CPGC model vs statistical baselines.

Compares bits/byte on a corpus of files:
  - Uniform (8 bits/byte — theoretical worst case)
  - Order-1 PPM
  - TinyLSTM (various hidden sizes)

Also attempts to call external compressors (zstd, lzma, brotli) if installed.

Usage:
    python benchmark.py --file path/to/enwik8 --steps 100000
    python benchmark.py --dir path/to/silesia/
"""

import argparse
import math
import os
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

from model_arch import TinyLSTM, Order1Model, synthetic_text


def bits_per_byte_lstm(data: bytes, hidden: int, lr: float, steps: int,
                        device: torch.device) -> float:
    model = TinyLSTM(hidden=hidden, embed_dim=hidden).to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    model.train()
    model.reset_state(device=device)
    n = min(len(data), steps + 1)
    total_bits = 0.0
    prev_t = torch.tensor([0], dtype=torch.long, device=device)
    for i in range(n - 1):
        actual = data[i + 1]
        probs  = model(prev_t)
        total_bits += -math.log2(max(probs[actual].item(), 1e-30))
        loss = -torch.log(probs[actual] + 1e-30)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.h = model.h.detach()
        model.c = model.c.detach()
        prev_t = torch.tensor([actual], dtype=torch.long, device=device)
    return total_bits / max(n - 1, 1)


def bits_per_byte_order1(data: bytes) -> float:
    m = Order1Model()
    total_bits = 0.0
    prev = 0
    for b in data:
        p = m.predict(prev)
        total_bits += -math.log2(max(p[b], 1e-30))
        m.update(prev, b)
        prev = b
    return total_bits / max(len(data) - 1, 1)


def compressed_bpb_external(tool: str, data: bytes) -> float | None:
    """Attempt to compress data with an external tool and return bits/byte."""
    cmds = {
        "zstd":  ["zstd", "-19", "-q", "-o", "{out}", "{inp}"],
        "lzma":  ["lzma", "-9", "-k", "{inp}", "--stdout"],
        "brotli": ["brotli", "-q", "11", "-o", "{out}", "{inp}"],
        "gzip":  ["gzip", "-9", "-c", "{inp}"],
    }
    if tool not in cmds:
        return None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as f:
            f.write(data)
            inp = f.name
        out = inp + ".compressed"
        cmd = [c.replace("{inp}", inp).replace("{out}", out) for c in cmds[tool]]
        result = subprocess.run(cmd, capture_output=True, timeout=60)
        if result.returncode == 0:
            if os.path.exists(out):
                size = os.path.getsize(out)
                os.unlink(out)
            else:
                size = len(result.stdout)
            os.unlink(inp)
            return (size * 8) / max(len(data) - 1, 1)
        os.unlink(inp)
    except Exception:
        pass
    return None


def benchmark_file(path: str, steps: int, hidden: int, lr: float,
                   device: torch.device):
    with open(path, "rb") as f:
        data = f.read(steps + 1)
    name = Path(path).name
    n    = len(data)
    print(f"\n=== {name} ({n:,} bytes, using {min(n, steps):,} for neural) ===")

    results = {}

    results["uniform"] = 8.0

    t0 = time.perf_counter()
    results["order-1"] = bits_per_byte_order1(data)
    print(f"  order-1:          {results['order-1']:.4f} bits/byte  ({time.perf_counter()-t0:.1f}s)")

    for h in [hidden]:
        key = f"lstm-{h}"
        t0 = time.perf_counter()
        results[key] = bits_per_byte_lstm(data, h, lr, steps, device)
        print(f"  {key:<16}  {results[key]:.4f} bits/byte  ({time.perf_counter()-t0:.1f}s)")

    for tool in ["zstd", "gzip", "lzma", "brotli"]:
        bpb = compressed_bpb_external(tool, data)
        if bpb is not None:
            results[tool] = bpb
            print(f"  {tool:<16}  {results[tool]:.4f} bits/byte")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file",    type=str,   default=None)
    parser.add_argument("--dir",     type=str,   default=None)
    parser.add_argument("--hidden",  type=int,   default=64)
    parser.add_argument("--lr",      type=float, default=0.005)
    parser.add_argument("--steps",   type=int,   default=100_000)
    parser.add_argument("--cpu",     action="store_true")
    args = parser.parse_args()

    device = torch.device("cpu") if args.cpu or not torch.cuda.is_available() else torch.device("cuda")
    print(f"Device: {device}")

    files = []
    if args.file:
        files.append(args.file)
    if args.dir:
        files.extend(str(p) for p in Path(args.dir).iterdir() if p.is_file())
    if not files:
        print("No --file or --dir specified. Running on synthetic data.")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as f:
            f.write(synthetic_text(args.steps + 1))
            files.append(f.name)

    all_results = {}
    for path in files:
        try:
            all_results[path] = benchmark_file(path, args.steps, args.hidden, args.lr, device)
        except Exception as e:
            print(f"  Error on {path}: {e}")

    print("\n=== Summary ===")
    print(f"{'File':<30}  {'order-1':>8}  {'lstm':>8}")
    for path, res in all_results.items():
        name = Path(path).name[:30]
        o1   = res.get("order-1", float("nan"))
        lstm = res.get(f"lstm-{args.hidden}", float("nan"))
        print(f"{name:<30}  {o1:>8.4f}  {lstm:>8.4f}")


if __name__ == "__main__":
    main()
