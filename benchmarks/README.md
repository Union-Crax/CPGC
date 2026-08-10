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

1. **512 MiB segments with 2^26 tables, together.** The one remaining
   memory-side question, and it needs a machine that can hold ~15 GB per
   segment. Note this is *not* the "several percent" the file used to promise:
   the clamp has since been swept on its own and is worth about 0.4% at 2^26.
   What is unmeasured is the combined change, because window has been the
   stronger lever and 512 MiB falls in the gap between the last size that helped
   (256 MiB) and the first that hurt (1 GB). See
   [How much is actually left in it](#how-much-is-actually-left-in-it).
2. **Modelling, not memory.** With the table lever measured and small, closing
   more of the 7.8% gap to zpaq means new or better models rather than a bigger
   machine. `MODEL_CAP`'s open tier and the profile clamp have both been swept
   and are documented below; neither has much left.

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
support, not because the curve had flattened.

### How much is actually left in it

This used to say "should be worth several percent". It is not: the clamp has
now been swept, and the answer is much smaller. Holding the window at the
256 MiB level 9 runs and moving only the ceiling:

| Profile clamp | Compressed | bpc | vs current |
|---:|---:|---:|---:|
| 2^23 | 46,616,910 | 1.3893 | +1.38% |
| 2^24 | 46,249,709 | 1.3783 | +0.59% |
| **2^25** (current) | **45,980,493** | **1.3703** | — |

Successive doublings return 0.79% then 0.59%, a ratio of about 0.75 each time.
Extrapolating that decay, **2^26 is worth roughly 0.4% for its extra 6 GiB per
segment**, and the entire remaining table lever — unlimited memory, same model —
converges to somewhere near 1.7%. Worth having on a big machine; not worth
reorganising the project around, and nothing like the figure this file carried
on reasoning alone.

Two honest caveats in the other direction. This sweep isolates table size at a
fixed window, so it does not measure the configuration a bigger machine would
actually run, which is 512 MiB segments *and* 2^26 tables together. Window has
been the stronger lever throughout — at a fixed 2^25 clamp, one 256 MiB segment
beats two 128 MiB ones by 3.9% — so the combined change could be worth
noticeably more than 0.4%. But it could also be worth less than nothing: a
single 1 GB segment at this clamp came out 3.66% *worse* than splitting, so the
window turns against you somewhere between 256 MiB and 1 GB, and 512 MiB sits in
that unmeasured gap. That one run is the experiment worth doing on hardware that
can hold it — not the clamp on its own.

The practical conclusion is that memory is no longer the interesting frontier
here. Getting materially closer to zpaq (142,252,605, about 7.8% below the
current result) means modelling work, not a bigger box.

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

### What the two mixer inputs per model are worth

Every bit-history model feeds the mixer twice: a learned state map, and a fixed
closed-form (Krichevsky–Trofimov) estimate of the same packed state. The second
looks redundant — it learns nothing — so it was tested two ways on the 256 MiB
segment, against 45,980,493:

| Second input | Compressed | vs current |
|---|---:|---:|
| closed-form estimate (current) | 45,980,493 | — |
| dropped entirely | 46,221,588 | +0.52% |
| a fast-adapting second state map | 49,340,242 | +7.31% |

Both alternatives are worse, and the reasons are worth keeping.

Dropping it costs half a percent, so it is carrying real information rather than
padding. That also disposes of a tempting generalisation: after five weak models
made things 0.85% worse, it was reasonable to think mixer *width* was itself the
problem and pruning would help. It is not — a good input pays for its width. The
five models failed because they were weak, not because they were extra.

The fast state map is much worse, and structurally so. A state map is shared
across *all* of a model's contexts, indexed only by the packed state and the
nibble depth. Adapting it quickly therefore does not track "this context
lately", it tracks whichever unrelated context happened to visit that state
last. The count-adaptive schedule is not a tunable there; it is what makes a
shared map meaningful at all.

**Measure at the segment size you ship, not on a slice.** This has now bitten
twice, in two unrelated places, and both times the slice pointed the wrong way.

The second case was a round of five extra models — indirect word, element x
word, numeric run, a three-byte skip-gram, and a second match model anchored on
a 16-byte suffix. On the first 16 MiB of enwik9 they were collectively worth
-0.18%, every one of them positive. On the 256 MiB segment level 9 actually
runs, the same five came out **+0.85% worse** (46,372,935 against 45,980,493).
They were reverted.

The likely mechanism is worth knowing before adding models: every input widens
the first-layer weight rows, and the mixer has to learn to suppress the ones
that say nothing. On a short segment the tables are far larger than the context
population, so a weak model costs almost nothing and its occasional hits are
free. On a long one the mixer's own capacity is the scarce resource, and
diluting it costs more than a marginal model returns. A model has to be *good*,
not merely non-negative, to survive at level 9.

So: screen on 16 MiB by all means, but confirm anything you intend to keep on a
256 MiB segment before believing it. A screen run is 3 minutes and a
confirmation is 35.

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
