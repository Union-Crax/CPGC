# CPGC

CPGC is an experimental, lossless general-purpose compressor built around the
**CPGC-NX** bit-level context-mixing engine. It includes a command-line tool, a
native desktop archive browser, solid directory archives, parallel compression,
and CRC-32 verification.

CPGC is optimized for compression ratio, especially on text. It is not intended
to replace fast codecs such as zstd or gzip when latency is the main concern.

## Highlights

- Bit-level context mixing with hashed byte, word, sparse, stride, indirect,
  and long-match models
- Online two-layer logistic mixer, adaptive probability maps, and a binary
  arithmetic coder
- Parallel independent segments for large inputs
- Turbo profiles at levels 1–3 and larger-memory profiles at levels 7–9
- Adaptive text dictionary and reversible structured-data transforms
- Detection and passthrough of incompressible regions
- Single-file `.cpgc` and solid multi-file archive support
- Native GUI and optional Windows Explorer integration
- CRC-32 verification on every decoded archive

The encoder and decoder update the same model in lockstep, so model state is not
stored in the archive. The archive records its segment and model profile, making
decoding independent of the machine's CPU count and SIMD support.

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

[enwik8](https://mattmahoney.net/dc/textdata.html) is the first 100,000,000
bytes of the English Wikipedia dump and a standard text-compression benchmark.
Every CPGC v10 archive below was decompressed and CRC-verified.

![enwik8 compressed size vs other tools](benchmarks/enwik8_sizes.png)

At level 9, CPGC produces **19,320,647 bytes (1.546 bits/byte)**. That is 22%
smaller than xz `-9e`, 24% smaller than zstd `-22`, 25% smaller than brotli
`-q11`, 9% smaller than 7-Zip's PPMd, and 3.8% smaller than CPGC v9. Research
compressors such as zpaq, PAQ8, cmix, and nncp still achieve better ratios, at
substantially higher runtime cost.

#### All nine levels

![CPGC level sweep on enwik8](benchmarks/enwik8_tradeoff.png)

| Level | Compressed size | Bits/byte | Compress | Decompress | Round-trip |
|---:|---:|---:|---:|---:|:---|
| 1 | 23,544,702 B | 1.884 | 27 s | 26 s | Verified |
| 2 | 22,735,441 B | 1.819 | 28 s | 25 s | Verified |
| 3 | 22,068,868 B | 1.766 | 30 s | 29 s | Verified |
| 4 | 20,903,387 B | 1.672 | 113 s | 115 s | Verified |
| 5 | 20,471,793 B | 1.638 | 149 s | 155 s | Verified |
| 6 | 20,203,405 B | 1.616 | 156 s | 173 s | Verified |
| 7 | 19,382,850 B | 1.551 | 358 s | 357 s | Verified |
| 8 | 19,320,647 B | 1.546 | 380 s | 380 s | Verified |
| 9 | **19,320,647 B** | **1.546** | 379 s | 376 s | Verified |

These measurements used a four-core container. Levels 8 and 9 currently
produce identical archives. Turbo level 1 compressed faster than xz `-9e` in
this environment (27 seconds versus 138 seconds) while producing a smaller
archive.

### enwik9

[enwik9](https://mattmahoney.net/dc/textdata.html) is the first 1,000,000,000
bytes of the same dump and is used by the Large Text Compression Benchmark and
the Hutter Prize. Every run was decompressed and CRC-verified.

![enwik9 compressed size vs other tools](benchmarks/enwik9_sizes.png)

| Level | Compressed size | Bits/byte | Compress | Decompress | Round-trip |
|---:|---:|---:|---:|---:|:---|
| 1 | 205,742,664 B | 1.646 | 5 min | 4 min | Verified |
| 3 | 192,130,481 B | 1.537 | 4 min | 4 min | Verified |
| 5 | 178,844,027 B | 1.431 | 17 min | 18 min | Verified |
| 9 | **172,426,003 B** | **1.379** | 38 min | 36 min | Verified |

These measurements used the same four-core container. The level 9 run was
capped at three workers so the extra-large models fit within 15 GB of RAM. Its
172,426,003-byte output is 12.6% smaller than xz `-9e`, 20% smaller than zstd
`-22`, and 3.7% smaller than 7-Zip's PPMd on the reference ranking. The default
level 5 also slightly beats PPMd's best reported size.

Full measurements and chart-generation scripts are in [`benchmarks/`](benchmarks/):

- [`results.csv`](benchmarks/results.csv) — complete enwik8 level sweep
- [`enwik9_results.csv`](benchmarks/enwik9_results.csv) — enwik9 results
- [`make_charts.py`](benchmarks/make_charts.py) — reproducible charts
- [`run_bench.sh`](benchmarks/run_bench.sh) — benchmark runner

## Project status

CPGC is experimental and its archive format is still evolving. The current
decoder accepts format version 10 archives; retain a matching binary for older
archives. For important data, keep an independent copy and use `cpgc verify`
after compression.

Run the test suite with:

```sh
cargo test --release --features gui
```
