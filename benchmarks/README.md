# Running the benchmarks

Everything here reproduces the numbers in the top-level README. The engine is
deterministic: the same input, level and build always produce the same bytes,
on any machine and whatever the core count or SIMD support. Only *timings* are
machine-dependent.

## Handoff: where this stands

**The benchmark table is complete and no longer the bottleneck.** As of
2026-08-08 every published enwik9 level has been measured and round-trip
verified against the current engine, so a change to the model can be judged
against a full baseline instead of a single point. Numbers are under
[Current results](#current-results).

Determinism is now tested rather than asserted: enwik8 -9 and enwik9 -9 were
re-run on a 16-core / 32 GB Windows desktop and came out byte-identical to
figures first measured on a 4-core / 15 GB Linux container — different OS,
different core count, different worker scheduling, same output. If you change
the model and the two anchors below stop matching, that is intentional and the
whole table needs re-measuring; if you change *scheduling* and they stop
matching, something is wrong.

| Anchor | Bytes |
|---|---:|
| enwik8 -9 | 18,122,756 |
| enwik9 -9 | 153,298,285 |

**What is still open, in the order I would try it:**

1. **Table size** — the one known lever with several percent in it and no
   modelling work. See [The open lever](#the-open-lever-table-size). It was not
   run this pass purely for want of RAM: the 2^26 profile needs ~15 GB free per
   worker and the machine had 14.6 GB. Read that section before starting — the
   memory estimate it used to carry was low by a third, and there is a much
   cheaper variant of the same experiment worth doing first.
2. **`MODEL_CAP`'s open tier.** Nineteen of the thirty models never reach the
   profile clamp, so the global ceiling is not what binds them — but thirteen of
   those nineteen are capped by arithmetic, not judgement, and raising them buys
   nothing. Only six are open-ended. See [The open lever](#the-open-lever-table-size).

`CPGC_MIX_LR` at 256 MiB is **not** open — it was measured at exactly that
segment size, three rates, before level 9 was set there. See the rate section
at the bottom; 3 is confirmed, and both neighbours are much worse.

Two practical notes for whoever picks this up. Levels 8 and 9 on enwik9 run for
hours — budget most of a day for a full re-measure, and split the level list
across invocations so a crash does not cost the lot. And `max_workers` reads
free memory to decide concurrency, so closing other applications genuinely
changes runtime: level 8 took 56 minutes with two workers where one would have
taken ~90.

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

The level list can be split across several invocations — pass a different `CSV=`
each time and concatenate — which is worth doing for enwik9, where levels 8 and
9 run for hours. On Windows use `python`, and set `PYTHONIOENCODING=utf-8` or
the ✓ in the printed tables will raise `UnicodeEncodeError` on a cp1252 console
after the charts have already been written.

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
| 8 | 64 MiB | ~5.2 GB |
| 9 | 256 MiB | ~8.6 GB |

Those two figures are `model_bytes` evaluated at the level's segment size, and
they check out against what the process actually holds: level 9 on enwik8 sat at
8.3 GB resident against a predicted 8.57 GiB, and two level-8 workers at 11.2 GB
against a predicted 10.3 GiB plus the 1 GB input. Trust `model_bytes` when
sizing a new profile — it is accurate to a few percent.

CPGC works out how many segments it can afford to run at once from the model
size and how much memory the machine reports free — `MemAvailable` on Linux,
`GlobalMemoryStatusEx`'s `ullAvailPhys` on Windows (`max_workers` in
`src/cm/mod.rs`) — so this is handled automatically and you do **not** need
`RAYON_NUM_THREADS`. On a machine with more memory it simply uses more cores
for identical output.

It budgets two thirds of what is free, so the worker count steps at awkward
places: level 8's model measures ~5.2 GB, so a second worker needs about
16 GB free, and level 9's ~8.6 GB model needs about 26 GB for a second. Closing
a few applications can halve a run.

Reference timings from the 4-core, 15 GB container the published numbers were
first measured on: enwik8 level 9 took 784 s to compress and 786 s to verify;
enwik9 level 9 took 124 min and 128 min. On the 16-core / 32 GB desktop the
current `enwik9_results.csv` came from, level 9 still ran one segment at a time
and took 107 min and 113 min. A machine that can hold all four enwik9 segments
at once — about 51 GB free, given the two-thirds budget — should finish level 9
in roughly a quarter of that.

## Current results

enwik9, every level round-trip verified:

| Level | Size | bpc | Compress | Decompress |
|---:|---:|---:|---:|---:|
| 1 | 205,528,124 | 1.644 | 2 min | 2 min |
| 3 | 191,638,802 | 1.533 | 3 min | 2 min |
| 5 | 172,544,182 | 1.380 | 14 min | 12 min |
| 8 | 158,556,042 | 1.268 | 56 min | 56 min |
| 9 | 153,298,285 | 1.226 | 107 min | 113 min |

enwik8 level 9 is 18,122,756 bytes (1.450 bpc); `results.csv` has all nine
levels.

Level 5 already beats `xz -9e` (197,331,816) by 12.6% on enwik9, and the top of
the range is where the segment-size lever shows up: level 8's 64 MiB segments
give up 3.4% against level 9's 256 MiB ones on the same model.

**Two machines are represented, so compare sizes across them but not times.**
`results.csv` (enwik8) holds the original 4-core / 15 GB container timings.
`enwik9_results.csv` was re-measured in full on a 16-core / 32 GB Windows
desktop, where level 8 got two concurrent workers and level 9 one. Both enwik8
-9 and enwik9 -9 reproduced byte-for-byte against the container's figures —
different OS, different core count, identical output, which is the determinism
claim above doing its job.

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

Raising the ceiling to 2^26 with a matching 512 MiB window (`seg_size_for_level`)
costs **about 14.8 GiB of model per segment, not the ~11 GB this file used to
estimate.** That is `model_bytes` evaluated for `plus == 3` at `n == 2^29`,
computed rather than measured, but by the same method that predicted the two
footprints above to within a few percent. Budget one worker per ~15 GB of real
RAM; `max_workers` always schedules at least one segment however little memory
it finds, so a machine that cannot hold the model will thrash rather than
refuse. Both the ceiling and the window must move together — a wider window
against unchanged tables is the regression described above.

Worth knowing before you spend the memory: at `MEM_HUGE` on a 256 MiB segment
the clamp yields 25 bits, but only 11 of the 30 models are uncapped enough to
use it. The other 19 bind at `MODEL_CAP`, so raising the global ceiling buys
them nothing and essentially all of the extra 6 GiB goes to doubling those
eleven (orders 4–7, word, word-pair, orders 8/10/12/16, word trigram).

Those eleven cannot be reached more cheaply, though — their `MODEL_CAP` entries
are already 31, i.e. no cap at all, so the clamp is the *only* thing limiting
them and raising their caps selectively is a no-op. Buying them more table means
paying the full 6 GiB.

The 19 that do bind split in two, and the distinction is what matters:

* **Thirteen are arithmetic.** An order-2 context has exactly 65,536 values, a
  stride lane 65,536, the nesting context 16,384. Multiply by the 17 buckets a
  context touches and their caps (21, 21, 19) are the population, not a guess.
  Raising these cannot help — the extra table is unaddressable by construction.
* **Six are judgements** — order-3, indirect order-3 and order-4, case-folded
  order-3, the enclosing element and the line shape. All are open-ended in real
  text, and all sit at `CAP_OPEN`, currently 24, because that seemed like enough
  table to spend. They are grouped under that name in `src/cm/predictor.rs` and
  move as a unit via `CPGC_CAP_OPEN` under the `tune` feature.

  **This tier has been measured and it is not the lever.** On the first 256 MiB
  of enwik9 as a single segment — what level 9 actually runs — raising the whole
  tier by a bit is worth 2,286 bytes:

  | `CAP_OPEN` | Compressed | bpc |
  |---:|---:|---:|
  | 24 (current) | 45,980,493 | 1.3703 |
  | 25 | 45,978,207 | 1.3703 |

  That is 0.005% for 1.5 GiB per segment. These six caps were already adequate;
  raising them is not a cheap substitute for the clamp, it is simply no gain.
  The reasoning that made it look promising — "nineteen models bind here, so
  this is where the crowding is" — was wrong, and the thirteen arithmetic caps
  above are why: most of that nineteen could never have been short of table.

So all of the crowding sits in the eleven clamp-bound models, and buying them
more table costs the full 6 GiB with no cheaper path to the same place.

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

That re-check has been done for the size level 9 actually uses. On the first
256 MiB of enwik9 as a single segment — the exact configuration level 9 runs —
all three candidate rates were measured:

| `CPGC_MIX_LR` | Compressed |
|---:|---:|
| 4 | 53,352,426 |
| **3** | **45,980,493** |
| 2 | 48,252,748 |

Rate 3 wins by 4.9% over its nearer neighbour, so the schedule is confirmed at
the size that matters and this is not an open question. What has *not* been
checked is a segment larger than 256 MiB, which only becomes reachable if the
table ceiling moves — and if it does, the rate must be re-swept there, because
that is precisely the mistake this warning is about.
