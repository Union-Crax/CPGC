# CPGC — Contextual Predictive Graph Compression

An experimental general-purpose compressor that uses an online-learning LSTM
ensemble to predict byte probabilities, feeding a rANS entropy coder.

**Status: research prototype — does not yet beat zstd or LZMA.**

---

## What it actually does

CPGC is a **byte-level prediction engine**. For each incoming byte it predicts
a probability distribution over all 256 possible next bytes, then feeds that
distribution to a rANS entropy coder. Fewer bits are used for high-probability
bytes. Encoder and decoder run the identical model in lockstep, so the model
is never stored in the output.

The ensemble combines:

- **TinyLSTM** (64 hidden units, ~66K params) — online SGD, weights update every byte
- **Order-1/2/4 tables** — statistical byte-pair / triplet / 4-gram frequencies
- **Run-length model** — catches long runs of identical bytes
- **LZ match model** — catches repeated substrings
- **ContextMixer** — online-learned weights blend all model outputs

---

## Where it stands vs existing compressors

Based on experiments at 50k–200k steps on enwik8 (100 MB Wikipedia XML):

| Compressor | bits/byte | Encode speed |
|---|---|---|
| gzip -9 | ~3.5 bpb | ~50 MB/s |
| bzip2 -9 | ~3.2 bpb | ~12 MB/s |
| **CPGC (200k steps)** | **~2.95 bpb** | **~0.4 MB/s** |
| zstd -19 | ~2.8 bpb | ~8 MB/s |
| LZMA ultra | ~2.6 bpb | ~2 MB/s |
| CMIX / PAQ | ~1.6 bpb | ~0.01 MB/s |

CPGC beats gzip and bzip2 on ratio. It does not yet beat zstd or LZMA.
The encode speed (~0.4 MB/s) is ~1000× slower than zstd — the online SGD
forward+backward pass runs on every single byte.

### Why it's slow

The LSTM weight update is the bottleneck. 66K parameters × 1 forward + 1
backward pass × every byte = expensive. The current SIMD matmul helps but
the algorithmic cost is fundamental to online per-byte learning.

### What it would need to be competitive

- Multi-step BPTT (currently truncated to 1 step)
- Larger / better architecture — the hidden-size sweep showed diminishing
  returns above 128 units, so bigger is not the answer alone
- More statistical models in the mixer — PAQ uses hundreds; CPGC uses 6
- GPU-accelerated batched inference for practical speeds

---

## Architecture

```
Input → Content Analyzer → Transform Preprocessor → Adaptive Neural Predictor → rANS → Output
```

- **2-pass rANS** — encoder buffers symbols, encodes in reverse; decoder reads stream backwards
- **Transform preprocessor** (8 candidate ops, level ≥ 5) — reduces entropy on structured binary blocks before prediction
- **Content analyzer** — passthrough for already-compressed / high-entropy blocks

---

## Build

```sh
cargo build --release
```

Binary: `target/release/cpgc.exe`

---

## CLI Usage

### Compress a file

```sh
cpgc compress <input> <output.cpgc> [-l <level>]
```

`-l` / `--level`: 1–9, default 5 (reserved for future tuning — currently no-op).

```
path/to/file.txt → file.cpgc
        12345 bytes →       9876 bytes  (0.800 ratio)
  0.423 MB/s  (0.03s)
```

### Decompress

```sh
cpgc decompress <input.cpgc> <output>
```

```
file.cpgc → recovered.txt
        12345 bytes recovered  (0.401 MB/s, 0.03s)
```

### Show archive info

```sh
cpgc info <file.cpgc>
```

```
CPGC archive: "file.cpgc"
  version:        1
  original size:  12345 bytes
  compressed:     9876 bytes
  ratio:          0.8000
  bits/byte:      6.4000
```

### Benchmark a corpus directory

Compresses every file in a directory and prints a table:

```sh
cpgc bench corpus/
```

```
file                                        orig(B)      comp(B)    ratio       bpb     MB/s
------------------------------------------------------------------------------------------
enwik8                                   100000000    67000000   0.6700    5.3600    0.421
------------------------------------------------------------------------------------------
TOTAL                                    100000000    67000000   0.6700    5.3600
```

### List archive contents

```sh
cpgc list <archive.cpgc>
```

For a single-file archive this shows the info header. For a solid multi-file
archive (`"CPAS"` magic) it prints the file table without decompressing.

---

## Experiment Results

### Learning rate sweep (hidden=64, 50k steps on enwik8)

Best LR: **0.01** → 3.43 bits/byte

![LR sweep](experiments/lr_sweep.png)

### Ablation study (contribution of each model component)

| Config | bits/byte |
|--------|-----------|
| uniform | 8.000 |
| + order-1 | 4.414 |
| + order-2 | 4.472 *(hurts at 50k — table too sparse)* |
| + LSTM | 3.802 |

![Ablation](experiments/ablation.png)

### Hidden size sweep (50k steps, lr=0.005)

| hidden | bits/byte | params |
|--------|-----------|--------|
| 32 | 3.153 | 25K |
| **64** | **2.955** | **66K** ← chosen |
| 128 | 2.906 | 198K |
| 256 | 2.903 | 658K |

---

## Roadmap

- [x] int8 weight quantization — `quantize_snapshot()` for compact model storage (~4× smaller)
- [x] Transform preprocessor wired into codec (level ≥ 5, 8 candidate ops, per-block entropy gain check)
- [x] Content analyzer wired for incompressible-region passthrough
- [x] Solid multi-file archive (`cpgc compress dir/ archive.cpgc`, `cpgc list archive.cpgc`)
- [x] Criterion benchmarks (`cargo bench`)
- [ ] True int8 runtime inference (hot-path dequantize-on-the-fly for better L2 cache utilization)
- [ ] LR decay schedule (0.9999 per byte as recommended in plan)
- [ ] Full enwik8 benchmark table vs zstd / LZMA
