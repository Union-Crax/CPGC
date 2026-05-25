"""
ablation.py — Measure the contribution of each mixer component.

Adds components one at a time and measures the improvement in bits/byte:
  1. Uniform baseline (8 bits/byte)
  2. Order-1 only
  3. Order-1 + Order-2
  4. Order-1 + Order-2 + LSTM (64 hidden)
  5. Full mixer (all 6 components)

Usage:
    python ablation.py --file path/to/enwik8 --steps 50000
    python ablation.py --synthetic --steps 30000
"""

import argparse
import math
import time
import numpy as np
import torch
import torch.optim as optim

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

from model_arch import TinyLSTM, Order1Model, synthetic_text


ALPHA = 1.0  # Laplace smoothing

# ---------------------------------------------------------------------------
# Minimal Order-2 model
# ---------------------------------------------------------------------------

class Order2Model:
    def __init__(self):
        self.counts = {}  # (prev2, prev1) -> np.array[256]

    def predict(self, prev2: int, prev1: int) -> np.ndarray:
        key = (prev2, prev1)
        if key in self.counts:
            row = self.counts[key]
            return row / row.sum()
        return np.ones(256, dtype=np.float32) / 256.0

    def update(self, prev2: int, prev1: int, nxt: int):
        key = (prev2, prev1)
        if key not in self.counts:
            self.counts[key] = np.ones(256, dtype=np.float32) * ALPHA
        self.counts[key][nxt] += 1.0


# ---------------------------------------------------------------------------
# Geometric mixer
# ---------------------------------------------------------------------------

def blend(predictions: list, weights: list) -> np.ndarray:
    log_p = np.zeros(256, dtype=np.float64)
    for pred, w in zip(predictions, weights):
        log_p += w * np.log(np.maximum(pred, 1e-30))
    # Softmax re-normalize
    log_p -= log_p.max()
    p = np.exp(log_p)
    return p / p.sum()


# ---------------------------------------------------------------------------
# Ablation configurations
# ---------------------------------------------------------------------------

CONFIGS = {
    "uniform":           {"order1": False, "order2": False, "lstm": False},
    "order1":            {"order1": True,  "order2": False, "lstm": False},
    "order1+order2":     {"order1": True,  "order2": True,  "lstm": False},
    "order1+order2+lstm":{"order1": True,  "order2": True,  "lstm": True},
}


def run_config(data: bytes, cfg: dict, lr: float, steps: int,
               device: torch.device, hidden: int = 64) -> float:
    use_order1 = cfg["order1"]
    use_order2 = cfg["order2"]
    use_lstm   = cfg["lstm"]

    o1 = Order1Model() if use_order1 else None
    o2 = Order2Model() if use_order2 else None

    lstm_model = None
    optimizer  = None
    if use_lstm:
        lstm_model = TinyLSTM(hidden=hidden, embed_dim=hidden).to(device)
        lstm_model.train()
        lstm_model.reset_state(device=device)
        optimizer = optim.SGD(lstm_model.parameters(), lr=lr, momentum=0.9)

    n = min(len(data), steps + 1)
    total_bits = 0.0

    prev1 = 0
    prev2 = 0
    prev_t = torch.tensor([0], dtype=torch.long, device=device)

    for i in range(n - 1):
        actual = data[i + 1]
        preds  = []
        ws     = []

        if not use_order1 and not use_order2 and not use_lstm:
            p = np.ones(256, dtype=np.float32) / 256.0
        else:
            if use_order1:
                preds.append(o1.predict(prev1))
                ws.append(1.0)
            if use_order2:
                preds.append(o2.predict(prev2, prev1))
                ws.append(1.0)
            if use_lstm:
                with torch.no_grad():
                    lp = lstm_model(prev_t).cpu().numpy()
                preds.append(lp)
                ws.append(1.0)
            # Equal weights (no online mixer learning in ablation for clarity)
            total_w = sum(ws)
            ws = [w / total_w for w in ws]
            p = blend(preds, ws)

        total_bits += -math.log2(max(p[actual], 1e-30))

        # Update statistical models
        if use_order1:
            o1.update(prev1, actual)
        if use_order2:
            o2.update(prev2, prev1, actual)
        if use_lstm:
            actual_t = torch.tensor([actual], dtype=torch.long, device=device)
            probs_t  = lstm_model(prev_t)
            loss     = -torch.log(probs_t[actual] + 1e-30)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lstm_model.h = lstm_model.h.detach()
            lstm_model.c = lstm_model.c.detach()
            prev_t = actual_t

        prev2 = prev1
        prev1 = actual

    return total_bits / max(n - 1, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file",      type=str,   default=None)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--hidden",    type=int,   default=64)
    parser.add_argument("--lr",        type=float, default=0.005)
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

    print(f"\nAblation study (steps={args.steps:,}, hidden={args.hidden}, lr={args.lr}):\n")

    results = {}
    for name, cfg in CONFIGS.items():
        t0 = time.perf_counter()
        bpb = run_config(data, cfg, lr=args.lr, steps=args.steps, device=device, hidden=args.hidden)
        elapsed = time.perf_counter() - t0
        print(f"  {name:<28}  {bpb:.4f} bits/byte  ({elapsed:.1f}s)")
        results[name] = bpb

    print("\nComponent contributions:")
    names = list(results.keys())
    for i in range(1, len(names)):
        delta = results[names[i-1]] - results[names[i]]
        print(f"  {names[i-1]!s} → {names[i]!s}: -{delta:.4f} bits/byte")

    if HAS_MPL:
        labels = list(results.keys())
        values = list(results.values())
        plt.figure(figsize=(10, 5))
        bars = plt.bar(range(len(labels)), values)
        plt.xticks(range(len(labels)), labels, rotation=20, ha="right")
        plt.ylabel("bits/byte (lower is better)")
        plt.title(f"Ablation study (steps={args.steps:,})")
        plt.tight_layout()
        out = "ablation.png"
        plt.savefig(out)
        print(f"\nPlot saved to {out}")


if __name__ == "__main__":
    main()
