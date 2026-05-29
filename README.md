# CPGC — Contextual Predictive Graph Compression

A general-purpose compressor built on **CPGC-NX**, a bit-level context-mixing
predictor feeding a binary arithmetic coder.

**Status: on general text it beats gzip, bzip2 and xz/LZMA on ratio.** It does
not beat the heaviest research compressors (cmix/PAQ8), and on tiny files or
already-dense binaries bzip2/xz can still edge it.

---

## What it actually does

CPGC-NX codes the input **one bit at a time**. For every bit it predicts
`P(next bit == 1)`, mixes the predictions of several context models in the
logistic domain, refines the result, and feeds it to a binary arithmetic
coder. Encoder and decoder run the identical model in lockstep, so the model
is never stored in the output.

The predictor is a new *combination* (the bit-level context-mixing framework is
shared with the PAQ family, but the model below is specific to this codec):

- **Dual learning-rate counters** — each context slot stores a *fast* and a
  *slow* adaptive probability; both are fed to the mixer, so it can trust the
  fast estimate during local change and the slow one when stationary.
- **Verified long-match model** — a rolling-hash pointer into the decoded
  history finds the most recent occurrence of the current suffix, *verifies* it
  by extending backward (rejecting hash collisions) and measures the true match
  length, then forecasts the bit of the continuation with confidence that grows
  with that length. Long verified matches are predicted near-certainly, which
  is what captures long-range redundancy and drives structured data well below
  1 bpb.
- **Context models** at orders 0,1,2,3,4,6 plus a whitespace-delimited word
  model, hashed into input-sized tables.
- **Two-context mixing layer** — predictions are mixed by two weight sets
  (selected by the previous byte and by match-length) and averaged in the
  logistic domain, then trained online by gradient descent on coding loss.
- **Chained SSE** — two adaptive probability maps refine the mixed estimate.

Table sizes are derived deterministically from the byte count (which both
sides know), so small inputs stay cheap and large inputs get the full model
without ever desyncing encoder and decoder.

### Scaling to big archives (parallelism)

Inputs larger than 16 MiB are split into fixed-size **independent segments**
that are compressed and decompressed in parallel across every CPU core. The
segment size is a fixed constant (not derived from the core count), so an
archive written on a 4-core machine decodes identically on a 64-core one.
Segments are large enough that per-segment model warm-up costs a negligible
amount of ratio on realistic data, while throughput scales close to linearly
with cores — so on big archives CPGC-NX is not only smaller than xz but
**faster** than it too.

---

## Measured results

Compressed size in bytes (smaller is better), with verified lossless
round-trips. Reference tools at maximum setting (`-9`).

| file | type | orig | **CPGC-NX** | gzip -9 | bzip2 -9 | xz -9 |
|---|---|--:|--:|--:|--:|--:|
| real40.txt | **40 MB text** | 40,002,833 | **6,106,235** | 11,029,458 | 6,877,636 | 8,318,320 |
| big.txt | 9 MB text | 9,227,058 | **756,972** | 2,201,048 | 1,045,632 | 1,174,956 |
| english.txt | text | 243,242 | **24,739** | 41,726 | 26,796 | 36,904 |
| code.txt | source | 147,807 | **32,851** | 39,844 | 35,571 | 35,260 |
| realtext.txt | small prose | 34,706 | 7,986 | 9,048 | **7,678** | 8,164 |
| binary.bin | executable | 400,000 | 144,872 | 175,089 | 173,161 | **142,904** |
| random.bin | incompressible | 200,000 | 200,071 | 200,064 | 201,284 | 200,072 |

On the 40 MB sample CPGC-NX is **27% smaller than xz/LZMA, 45% smaller than
gzip, 13% smaller than bzip2 — and at 1.75 MB/s it compresses faster than xz**
(1.27 MB/s) by using all cores. Incompressible data is detected and passed
through, so there is no expansion blow-up.

### Honesty about the limits

- This is **not** magic and does not beat cmix/PAQ8-class compressors.
- On very small files there is little history to learn from, so a BWT
  compressor (bzip2) can win.
- On high-entropy binaries the gap to xz narrows or reverses.
- On *pathologically* redundant inputs (the same multi-MB block repeated many
  times) xz's 64 MB LZ window wins, because such repeats can straddle the
  segment boundaries CPGC compresses independently. Real archives rarely look
  like this; on varied data segmentation costs almost nothing.
- Single-segment throughput is ~0.4–0.8 MB/s (the cost of per-bit mixing);
  parallelism is what lifts large-archive throughput past xz.

The previous online-LSTM engine (which topped out around 2.95 bpb at ~0.4 MB/s)
still lives in `src/predictor/` and `src/ans/` for reference, but is no longer
on the compression path.

### Binary media & game files

Text is the easy case; structured binary is where naive byte models fall over.
Uncompressed media looks random at the byte level (16-bit audio, RGB pixels)
even though it is highly compressible by *stride*. CPGC-NX adds **sparse stride
models** (strides 2, 3, 4, 8) that predict each byte from the same lane of the
previous sample(s), and a **stride-aware incompressibility test** so structured
media reaches the compressor instead of being passed through.

| file | type | orig | **CPGC-NX** | gzip -9 | bzip2 -9 | xz -9 |
|---|---|--:|--:|--:|--:|--:|
| image.rgb | 24-bit image | 2,430,000 | **54,609** | 424,672 | 409,657 | 115,296 |
| exe.bin | executable | 1,124,888 | **372,582** | 492,925 | 463,361 | 398,536 |
| game.dat | record data | 1,920,000 | 644,815 | 1,070,121 | 793,477 | **640,812** |
| audio.pcm | 16-bit PCM | 2,400,000 | 965,128 | 2,311,622 | 2,079,406 | **596,752** |

CPGC-NX wins clearly on images and executables and ties on record data.
Already-compressed media (MP3, JPEG, H.264, PNG, ZIP'd game assets) is
high-entropy and is **passed through unchanged** — no wasted time, no
expansion. Raw PCM audio still trails xz (linear-predictive residuals are
LZMA's strong suit) but is no longer mishandled.

---

## Architecture

```
Input → Content Analyzer → Transform Preprocessor → CPGC-NX context mixer → Binary arithmetic coder → Output
```

- **CPGC-NX predictor** (`src/cm/`) — bit-level context mixing (see above)
- **Binary arithmetic coder** — carryless 32-bit range coder, codes one bit at
  a time given a 12-bit probability
- **Transform preprocessor** (8 candidate ops, level ≥ 5) — reduces entropy on structured binary blocks before prediction
- **Content analyzer** — passthrough for already-compressed / high-entropy blocks

---

## Download / install (no compiling)

Prebuilt binaries and a Windows installer are produced by GitHub Actions for
every tagged release (and as downloadable artifacts on every run). There are
**two separate downloads** — grab whichever you need:

- **`CPGC-Setup.exe`** (Windows) — a real installer for the desktop app
  (`cpgc-gui`): Start-Menu + desktop shortcuts, optional "add CLI to PATH".
- **`cpgc-cli-<os>`** — just the `cpgc` command-line tool.
- **`cpgc-gui-<os>`** — the desktop app on its own (macOS/Linux).

See the project's **Releases** page, or **Actions ▸ build** for per-commit
artifacts.

## Build from source

The CLI and GUI are **separate binaries**, so the CLI builds without any GUI
dependencies:

```sh
cargo build --release --bin cpgc                  # CLI only (lean)
cargo build --release --features gui --bin cpgc-gui   # desktop app
```

---

## GUI (native 7-Zip-style app)

`cpgc-gui` (run the binary, or `cpgc-gui /some/folder` to start there) opens a
real native window — not a browser:

- A **menu bar**, a **toolbar** (Up / Add / Info / Extract / Test), a file list
  with **Name / Size / Modified** columns, and a **status bar** showing the
  selection count, like a familiar archive manager.
- **Add** opens an "Add to Archive" dialog (name + level) and compresses the
  ticked files/folders into a `.cpgc` single-file or `.cpas` solid archive.
- **Open** an archive to browse its members, tick individual ones, and
  **Extract** them (or **Extract all**). **Test** verifies an archive.
- Every long operation runs on a background thread and can be **paused,
  resumed, or cancelled**, with a live progress bar and **throughput (MB/s)**.

Cross-platform (Windows/macOS/Linux); it needs a graphical desktop to run.

---

## Compression levels (CLI `-l`, or the GUI level selector)

`-l`/`--level` (1–9, default 5) trades speed for ratio by setting the parallel
**segment size**: lower levels use smaller segments (more cores, faster, a
little larger), higher levels use larger segments (better ratio, less
parallelism). The size is recorded in the archive, so decompression never needs
to know the level. Levels ≥ 5 also enable the transform pass. Example on 9 MB of
text:

| level | size | bits/byte | speed |
|--:|--:|--:|--:|
| 1 (fastest) | 889 KB | 0.77 | 1.08 MB/s |
| 3 | 794 KB | 0.69 | 0.73 MB/s |
| 5 (default) | 758 KB | 0.66 | 0.34 MB/s |
| 9 (best ratio) | 758 KB | 0.66 | 0.34 MB/s |

(Levels 5–9 coincide here because the file is smaller than one segment; they
diverge on larger archives.)

---

## CLI Usage

Subcommands have short aliases: `c` (compress), `x` (decompress), `t` (verify).

### Compress a file or directory

```sh
cpgc compress <input> [output.cpgc] [-l <level>]
```

The output path is optional — it defaults to `<input>.cpgc`. Pass a directory
to build a solid multi-file archive. `-l`/`--level` is 1–9 (default 5).

```
"file.txt" → "file.txt.cpgc"
        12345 bytes →       9876 bytes  (0.800 ratio)
  0.423 MB/s  (0.03s)
```

### Decompress / extract

```sh
cpgc decompress <input.cpgc> [output]
```

The output is optional — it defaults to the original name (`.cpgc` stripped).
Solid (`.cpas`) archives are unpacked into the output directory automatically.

```
"file.txt.cpgc" → "file.txt"
        12345 bytes recovered  (0.401 MB/s, 0.03s)
```

### Verify an archive

```sh
cpgc verify <archive>          # decodes in memory, writes nothing
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
