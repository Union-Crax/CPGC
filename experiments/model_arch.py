"""
model_arch.py — Phase 1 experiments for CPGC.

Implements and benchmarks the TinyLSTM online predictor in PyTorch.
Measures bits/byte on a user-supplied file (or generates synthetic data).

Usage:
    python model_arch.py --file path/to/enwik8 --hidden 64 --lr 0.005 --steps 100000
    python model_arch.py --synthetic --steps 50000

Output: bits/byte curve, final score, encode throughput estimate.
"""

import argparse
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ---------------------------------------------------------------------------
# TinyLSTM model
# ---------------------------------------------------------------------------

class TinyLSTM(nn.Module):
    """
    Online LSTM predictor.
    Input:  one byte (as integer index)
    Output: probability distribution over 256 byte values
    """
    def __init__(self, hidden: int = 64, embed_dim: int = 64):
        super().__init__()
        self.embed = nn.Embedding(256, embed_dim)
        self.lstm  = nn.LSTMCell(embed_dim, hidden)
        self.proj  = nn.Linear(hidden, 256)
        self.hidden_size = hidden
        self.h = None
        self.c = None

    def reset_state(self, batch_size: int = 1, device=None):
        dev = device or next(self.parameters()).device
        self.h = torch.zeros(batch_size, self.hidden_size, device=dev)
        self.c = torch.zeros(batch_size, self.hidden_size, device=dev)

    def forward(self, byte_idx: torch.Tensor):
        """
        byte_idx: LongTensor shape (1,) — the current byte
        Returns: FloatTensor shape (256,) — probability distribution
        """
        x = self.embed(byte_idx)          # (1, embed_dim)
        self.h, self.c = self.lstm(x, (self.h, self.c))
        logits = self.proj(self.h)         # (1, 256)
        return torch.softmax(logits[0], dim=-1)  # (256,)


# ---------------------------------------------------------------------------
# Order-1 statistical model (baseline)
# ---------------------------------------------------------------------------

class Order1Model:
    def __init__(self, alpha: float = 1.0):
        self.counts = np.ones((256, 256), dtype=np.float32)  # Laplace smoothing
        self.alpha  = alpha

    def predict(self, prev: int) -> np.ndarray:
        row = self.counts[prev]
        return row / row.sum()

    def update(self, prev: int, nxt: int):
        self.counts[prev, nxt] += 1.0


# ---------------------------------------------------------------------------
# Encoding loop
# ---------------------------------------------------------------------------

def encode_file(data: bytes, model: TinyLSTM, lr: float,
                device: torch.device, report_every: int = 10_000,
                max_steps: int = None) -> list[float]:
    """
    Run the online encoding loop.
    Returns a list of cumulative bits/byte snapshots.
    """
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    model.train()
    model.reset_state(device=device)

    n = len(data) if max_steps is None else min(len(data), max_steps)
    total_bits = 0.0
    snapshots  = []

    prev_byte = torch.tensor([0], dtype=torch.long, device=device)

    start = time.perf_counter()
    for i in range(n - 1):
        actual_byte = data[i + 1]
        actual_t    = torch.tensor([actual_byte], dtype=torch.long, device=device)

        # Forward
        probs = model(prev_byte)

        # Measure bits used for this byte: -log2(P(actual))
        p = probs[actual_byte].item()
        p = max(p, 1e-30)
        total_bits += -math.log2(p)

        if (i + 1) % report_every == 0:
            elapsed = time.perf_counter() - start
            bpb = total_bits / (i + 1)
            mb_s = (i + 1) / elapsed / 1e6
            print(f"  [{i+1:>8,}] {bpb:.4f} bits/byte  {mb_s:.3f} MB/s")
            snapshots.append(bpb)

        # Backward: cross-entropy on actual byte
        loss = -torch.log(probs[actual_byte] + 1e-30)
        optimizer.zero_grad()
        loss.backward(retain_graph=False)
        optimizer.step()

        # Detach state to avoid accumulating computation graph
        model.h = model.h.detach()
        model.c = model.c.detach()

        prev_byte = actual_t

    elapsed = time.perf_counter() - start
    final_bpb = total_bits / max(n - 1, 1)
    mb_s = (n - 1) / elapsed / 1e6
    print(f"\nFinal: {final_bpb:.4f} bits/byte  |  {mb_s:.3f} MB/s  |  {elapsed:.1f}s")
    return snapshots, final_bpb


# ---------------------------------------------------------------------------
# Hidden size sweep
# ---------------------------------------------------------------------------

def sweep_hidden_sizes(data: bytes, sizes: list[int], lr: float,
                       steps: int, device: torch.device):
    print("\n=== Hidden size sweep ===")
    results = {}
    for h in sizes:
        print(f"\n-- hidden={h} --")
        model = TinyLSTM(hidden=h, embed_dim=h).to(device)
        _, bpb = encode_file(data, model, lr=lr, device=device, max_steps=steps)
        results[h] = bpb
    print("\nSummary:")
    for h, bpb in sorted(results.items()):
        params = sum(p.numel() for p in TinyLSTM(hidden=h, embed_dim=h).parameters())
        print(f"  hidden={h:>3}  {bpb:.4f} bits/byte  ({params:,} params)")
    return results


# ---------------------------------------------------------------------------
# Synthetic data generators
# ---------------------------------------------------------------------------

def synthetic_text(n: int) -> bytes:
    """Simple Markov chain: alternating word-like patterns."""
    rng = np.random.default_rng(42)
    alphabet = list(b'abcdefghijklmnopqrstuvwxyz ')
    data = bytes(rng.choice(alphabet, size=n))
    return data

def synthetic_structured(n: int) -> bytes:
    """Structured binary: repeating 4-byte patterns with small delta."""
    pattern = np.arange(n, dtype=np.uint8) % 256
    noise   = np.random.default_rng(42).integers(0, 4, size=n, dtype=np.uint8)
    return bytes((pattern + noise) % 256)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="CPGC Phase 1: TinyLSTM experiments")
    parser.add_argument("--file",      type=str,   default=None,  help="Input file (e.g. enwik8)")
    parser.add_argument("--synthetic", action="store_true",       help="Use synthetic data")
    parser.add_argument("--hidden",    type=int,   default=64,    help="LSTM hidden size")
    parser.add_argument("--embed",     type=int,   default=64,    help="Embedding dim")
    parser.add_argument("--lr",        type=float, default=0.005, help="Learning rate")
    parser.add_argument("--steps",     type=int,   default=100_000, help="Max steps")
    parser.add_argument("--sweep",     action="store_true",       help="Sweep hidden sizes")
    parser.add_argument("--cpu",       action="store_true",       help="Force CPU")
    args = parser.parse_args()

    device = torch.device("cpu") if args.cpu or not torch.cuda.is_available() else torch.device("cuda")
    print(f"Device: {device}")

    # Load data
    if args.file:
        with open(args.file, "rb") as f:
            data = f.read(args.steps + 1)
        print(f"Loaded {len(data):,} bytes from {args.file}")
    elif args.synthetic:
        data = synthetic_text(args.steps + 1)
        print(f"Using synthetic text data ({len(data):,} bytes)")
    else:
        print("No --file or --synthetic specified. Using short synthetic demo.")
        data = synthetic_text(20_000)

    if args.sweep:
        sweep_hidden_sizes(data, [32, 64, 128, 256], lr=args.lr, steps=args.steps, device=device)
    else:
        model = TinyLSTM(hidden=args.hidden, embed_dim=args.embed).to(device)
        params = sum(p.numel() for p in model.parameters())
        print(f"Model: hidden={args.hidden}  embed={args.embed}  params={params:,}")
        encode_file(data, model, lr=args.lr, device=device,
                    report_every=max(1000, args.steps // 20), max_steps=args.steps)


if __name__ == "__main__":
    main()
