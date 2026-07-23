# CPGC

CPGC is an experimental, lossless compressor built on the **CPGC-NX** bit-level
context-mixing engine. It trades speed for ratio on text — it is not a
replacement for zstd or gzip when latency matters. It ships as a command-line
tool and a native archive-browser GUI, supports single-file `.cpgc` and solid
multi-file archives, and CRC-32-verifies every archive it decodes.

## How it works

The engine predicts each bit from ~26 context models — hashed byte contexts
(orders 2–16), word and word-pair models, sparse and stride contexts, indirect
models, and a long-match model — combined by a two-layer logistic mixer and
sharpened by a chained SSE stage before a binary arithmetic coder. Encoder and
decoder update the same model in lockstep, so no model state is stored in the
archive; it records only the segment size and model profile, so a file decodes
identically regardless of the machine's CPU count or SIMD support.

Large inputs are split into independent segments for parallel compression,
incompressible regions are detected and stored raw, and texty input can pass
through an adaptive word dictionary or reversible structured-data transforms.

## Install

Tagged releases publish binaries for Windows, macOS, and Linux. Windows releases
also include an installer.

[Download the latest release](https://github.com/Union-Crax/CPGC/releases/latest)

Available assets:

| Platform | Asset |
|---|---|
| Windows | `cpgc-x86_64-pc-windows-msvc.zip` |
| Windows installer | `CPGC-Setup.exe` |
| Linux | `cpgc-x86_64-unknown-linux-gnu.tar.gz` |
| macOS | `cpgc-x86_64-apple-darwin.tar.gz` |

## Build from source

Install the stable [Rust toolchain](https://rustup.rs/), clone the repository,
then build the binary you need:

```sh
# CLI
cargo build --release --bin cpgc

# Native GUI
cargo build --release --features gui --bin cpgc-gui
```

Built binaries are written to `target/release/`. On Linux, the GUI also needs
the X11/Wayland and OpenGL development libraries listed in
[the build workflow](.github/workflows/build.yml).

## CLI

```text
cpgc compress <input> [output] [--level <1-9>]
cpgc decompress <archive> [output]
cpgc verify <archive>
cpgc list <archive>
cpgc info <archive>
cpgc bench <corpus-directory>
```

`compress`, `decompress`, and `verify` also have the aliases `c`, `x`, and `t`.

### Common examples

```sh
# Creates notes.txt.cpgc at the default level (5)
cpgc compress notes.txt

# Choose an output path and compression level
cpgc compress notes.txt notes.cpgc --level 7

# Restore notes.txt from notes.txt.cpgc
cpgc decompress notes.txt.cpgc

# Pack and extract a directory as a solid archive
cpgc compress project/ project.cpas
cpgc decompress project.cpas restored-project/

# Decode and verify without writing output
cpgc verify project.cpas

# Inspect an archive
cpgc list project.cpas
cpgc info notes.cpgc
```

If no compression output is supplied, CPGC appends `.cpgc`. If no extraction
output is supplied, it strips `.cpgc` or `.cpas`; otherwise it appends `.out`.
Directory inputs are automatically stored as solid multi-file archives.

## Compression levels

Levels trade speed, parallelism, memory, and ratio. Level 5 is the default.

| Level | Segment size | Model | Memory profile | Block transforms |
|---:|---:|---|---|---|
| 1 | 1 MiB | Turbo + text dictionary | Standard | No |
| 2 | 2 MiB | Turbo + text dictionary | Standard | No |
| 3 | 4 MiB | Turbo + text dictionary | Standard | No |
| 4 | 8 MiB | Full | Standard | No |
| 5 | 16 MiB | Full | Standard | Yes |
| 6 | 32 MiB | Full | Standard | Yes |
| 7 | 64 MiB | Full | Big | Yes |
| 8–9 | 64 MiB | Full | Extra large | Yes |

High-entropy regions may be stored without context mixing at every level.
Levels 7–9 can require substantial memory on large files; use level 5 or 6 on
memory-constrained systems. Levels 8 and 9 currently use the same codec profile.

## Desktop GUI

Build with the `gui` feature, then run:

```sh
cpgc-gui
cpgc-gui /path/to/folder
cpgc-gui archive.cpgc
```

The GUI can browse folders and archives, create archives, extract selected or
all members, test integrity, show archive information, switch themes, and
pause, resume, or cancel long operations.

### Windows Explorer integration

From a stable installation path, run:

```sh
cpgc register
cpgc unregister
```

Registration is per-user under `HKCU` and does not require administrator
rights. It adds compression actions for files and folders plus open, extract,
and test actions for `.cpgc` and `.cpas` archives.

## The English Wikipedia benchmarks

### enwik8

[enwik8](https://mattmahoney.net/dc/textdata.html) is the first 100 MB of the
English Wikipedia dump, a standard text-compression benchmark. At level 9 CPGC
compresses it to **19,178,089 bytes (1.534 bpc)** — smaller than every
general-purpose codec below; the research compressors zpaq, PAQ8, and cmix
still lead. Every archive was round-trip decompressed and CRC-verified.

![enwik8 compressed size vs other tools](benchmarks/enwik8_sizes.png)

The nine levels trade compress time for ratio:

![CPGC level sweep on enwik8](benchmarks/enwik8_tradeoff.png)

| Level | Compressed size | Bits/byte | Compress | Decompress |
|---:|---:|---:|---:|---:|
| 1 | 23,539,435 B | 1.883 | 28 s | 25 s |
| 2 | 22,743,019 B | 1.819 | 28 s | 25 s |
| 3 | 22,065,155 B | 1.765 | 29 s | 28 s |
| 4 | 20,818,067 B | 1.665 | 123 s | 126 s |
| 5 | 20,388,399 B | 1.631 | 161 s | 164 s |
| 6 | 20,140,482 B | 1.611 | 162 s | 164 s |
| 7 | 19,249,638 B | 1.540 | 376 s | 367 s |
| 8 | 19,178,089 B | 1.534 | 423 s | 415 s |
| 9 | **19,178,089 B** | **1.534** | 410 s | 438 s |

Measured on a four-core container. Levels 8 and 9 currently produce identical
archives.

### enwik9

[enwik9](https://mattmahoney.net/dc/textdata.html) is the first 1 GB of the same
dump — the Large Text Compression Benchmark and Hutter Prize file. At level 9
CPGC reaches **163,890,252 bytes (1.311 bpc)**. Every archive was round-trip
decompressed and CRC-verified.

![enwik9 compressed size vs other tools](benchmarks/enwik9_sizes.png)

| Level | Compressed size | Bits/byte | Compress | Decompress |
|---:|---:|---:|---:|---:|
| 1 | 205,675,118 B | 1.645 | 5 min | 4 min |
| 3 | 192,017,370 B | 1.536 | 4 min | 4 min |
| 5 | 176,029,194 B | 1.408 | 21 min | 22 min |
| 9 | **163,890,252 B** | **1.311** | 39 min | 40 min |

Same four-core container; level 9 was capped at three workers to fit its models
within 15 GB of RAM.

Full measurements and chart-generation scripts are in [`benchmarks/`](benchmarks/):

- [`results.csv`](benchmarks/results.csv) — complete enwik8 level sweep
- [`enwik9_results.csv`](benchmarks/enwik9_results.csv) — enwik9 results
- [`make_charts.py`](benchmarks/make_charts.py) — reproducible charts
- [`run_bench.sh`](benchmarks/run_bench.sh) — benchmark runner

## Project status

CPGC is experimental and its archive format is still evolving. The current
decoder accepts format version 11 archives; retain a matching binary for older
archives. For important data, keep an independent copy and use `cpgc verify`
after compression.

Run the test suite with:

```sh
cargo test --release --features gui
```
