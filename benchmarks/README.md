# Running the benchmarks

Everything here reproduces the numbers in the top-level README. The engine is
deterministic: the same input, level and build always produce the same bytes,
on any machine and whatever the core count or SIMD support. Only *timings* are
machine-dependent.

## Getting the data

```sh
curl -O https://mattmahoney.net/dc/enwik8.zip && unzip enwik8.zip   # 100 MB
curl -O https://mattmahoney.net/dc/enwik9.zip && unzip enwik9.zip   # 1 GB
```

## Running

```sh
cargo build --release --bin cpgc

# enwik8: all nine levels + gzip/bzip2/xz references -> results.csv
DATA=/path/to/enwik8 ./run_bench.sh

# enwik9: chosen levels -> enwik9_results.csv
DATA=/path/to/enwik9 LEVELS="1 3 5 8 9" ./run_bench9.sh

# charts + README tables from whatever CSVs exist
python3 make_charts.py          # needs matplotlib
```

Both scripts compress, then `verify` (decode and CRC-check) every archive, and
record the result — a `verified` column of `0` means the level failed its round
trip and the number must not be published.

## What each level costs

Level 9 uses 256 MiB segments and the maximum memory profile; levels 7 and 8
use 64 MiB segments with smaller tables. Approximate model size **per segment
in flight**:

| Level | Segment | Model per worker |
|---:|---:|---:|
| 5–6 | 16–32 MiB | tens of MB |
| 7 | 64 MiB | ~2.5 GB |
| 8 | 64 MiB | ~5 GB |
| 9 | 256 MiB | ~8 GB |

CPGC works out how many segments it can afford to run at once from the model
size and the machine's `MemAvailable` (`max_workers` in `src/cm/mod.rs`), so
this is handled automatically — you do **not** need `RAYON_NUM_THREADS`. On a
machine with more memory it simply uses more cores for identical output.

Reference timings from the 4-core, 15 GB container the published numbers were
measured on: enwik8 level 9 took 784 s to compress and 786 s to verify; enwik9
level 9 took 124 min and 128 min. A machine that can run all four enwik9
segments concurrently should finish level 9 in roughly a quarter of that.

## Current results

| File | Level | Size | bpc |
|---|---:|---:|---:|
| enwik8 | 9 | 18,122,756 | 1.450 |
| enwik9 | 9 | 153,298,285 | 1.226 |

`enwik9_results.csv` currently holds level 9 only — levels 1/3/5/8 have not been
re-measured against the current engine, and the older figures for them were
dropped rather than left in place implying they were current. Re-running
`run_bench9.sh` with `LEVELS="1 3 5 8"` fills the table in.

## The open lever: table size

Segment size is the dominant ratio lever on large text, but it only pays while
the hashed tables can hold a segment's contexts. Measured over the first 256 MB
of enwik9, every doubling of the window paid:

| Configuration | Size |
|---|---:|
| 4 × 64 MiB | 48,491,970 |
| 2 × 128 MiB | 47,851,150 |
| 1 × 256 MB | 45,980,493 |

and then it reversed hard. The tables cap at 2^25 buckets whatever the segment
size, so a 1 GB segment gets 531 bytes of input per context slot where a 256 MB
one gets 136. Compressed as a single 1 GB segment, enwik9 came out at
168,708,258 bytes — worse than splitting it.

So level 9 sits at 256 MiB because that is the widest window *these tables*
support, not because the curve had flattened. **On a machine with more memory,
raising the table ceiling should be worth several percent with no modelling
change.** The ceiling is one expression, `model_bits` in
`src/cm/predictor.rs`:

```rust
raw_bits(n).clamp(11, 23 + plus)   // plus == 2 at MEM_HUGE, so 2^25
```

Raising `MEM_HUGE` to 2^26 costs roughly 11 GB of model per segment; the
matching window is then 512 MiB, set in `seg_size_for_level`. Both must move
together — a wider window against unchanged tables is the regression described
above. `model_bytes` reports the exact allocation for a given segment size and
profile, and `max_workers` will scale the pool to whatever the machine has.

Note that this changes the format: the profile byte is recorded in the payload,
so archives written with a larger profile need a decoder that knows it. Bump
`VERSION` in `src/codec.rs` if the bitstream changes.

## Measuring model changes quickly

A full enwik8 level-9 run takes ~13 minutes, which is too slow to iterate on.
The `probe` example encodes a file as a single segment and prints rolling
bits-per-byte every MiB:

```sh
cargo build --release --example probe
./target/release/examples/probe <file> <mem-profile 0-3>
```

A 16 MiB slice of enwik8 at profile 2 runs in about two minutes and tracks the
full-file result closely enough to rank variants.

For A/B testing without rebuilding, the `tune` feature exposes model
hyper-parameters as environment overrides:

```sh
cargo build --release --features tune --example probe
CPGC_MIX_LR=3 ./target/release/examples/probe file 2
```

Available: `CPGC_MIX_LR`, `CPGC_MIX_ROWS`, `CPGC_NBH`, `CPGC_WAYS`,
`CPGC_SM_DEPTH`, `CPGC_FAST_LEN`. The feature is off in release builds, so the
shipped bitstream never depends on the environment.

**One warning from experience.** `CPGC_MIX_LR` is the mixer learning rate, and
its right value falls as the segment grows — 4 for segments up to 64 MiB, 3
above. Tuning it on small slices and applying the result to large segments costs
5.3% on a 128 MiB segment, and it masks the segment-size lever entirely: at the
wrong rate, one 128 MiB segment looks 2.7% *worse* than two 64 MiB ones when it
is actually 2.7% better. Any model parameter tuned on a 16 MiB slice should be
re-checked at the segment size it will actually run at.
