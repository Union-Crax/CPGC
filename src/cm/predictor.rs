//! Bit-level context-mixing predictor for the CPGC-NX engine (v13).
//!
//! Produces `P(next bit == 1)` as a 12-bit probability. The architecture is a
//! new *combination* tuned for this codec rather than a port of any single
//! existing compressor:
//!
//! * **Universal bit-history states.** Every hashed context slot is a single
//!   packed byte: capped, mutually-discounting counts of observed 0s and 1s.
//!   Incrementing one count decays the other, so the state encodes *both*
//!   the evidence and its recency. A learned per-model state map converts
//!   the state to a probability (count-adaptive rate), and a closed-form
//!   direct estimate of the same state is fed to the mixer alongside it —
//!   recovering the fast/slow dual-view of v6 at one sixth the memory.
//! * **Nibble-bucketed, checksummed hash tables.** Context slots are grouped
//!   into 16-byte buckets holding the full 15-node subtree of one nibble:
//!   one hash lookup and one cache line serve four bits, where v6 took a
//!   fresh random lookup per bit. A one-byte checksum detects collisions and
//!   an n-way replacement policy evicts the least-established bucket,
//!   so colliding contexts no longer silently corrupt each other. All
//!   candidate lines are prefetched at the nibble boundary so the misses
//!   overlap instead of serialising.
//! * **Dual long-match model.** Two rolling hashes (8-byte and 4-byte suffix)
//!   point at the most recent place the current suffix occurred — the longer
//!   hash is preferred so seeds start from more reliable anchors — and the
//!   predictor forecasts the *bit* of the historical continuation with
//!   confidence that grows with verified match length.
//! * **Markup models.** Structured text — XML, HTML, wiki markup, source —
//!   is predicted as much by *where* a byte sits as by what precedes it, so
//!   four models track state no order-n context can see: the innermost open
//!   element, the innermost unclosed bracket and its depth, the column and
//!   shape of the current line, and the word trigram. On enwik8 these four
//!   are worth 1.3% on their own.
//! * **Two-layer logistic mixer.** A first layer holds six independently
//!   context-selected weight vectors (by previous byte, by a hashed order-3
//!   context, by match-length bucket, by the partial byte being decoded, by a
//!   hashed order-6 context, and by the current word hash); a small learned
//!   second layer selected by (previous-byte class, match length, bit
//!   position) combines their stretched outputs, trained online by gradient
//!   descent.
//! * **Cardinality-budgeted tables.** Each model's hash table is capped at
//!   the size its context population can actually fill, counting the 17
//!   buckets a context touches across nibble paths. The memory that frees is
//!   spent on the high-order, word and indirect models, which on a
//!   segment-sized-as-the-whole-file run see hundreds of millions of contexts.
//! * **Chained SSE.** Four adaptive probability maps in an increasing-order
//!   ladder (keyed by the partial byte, an order-2 context, an order-3 hash,
//!   and an order-6 hash), each nudging the running estimate a quarter of the
//!   way toward its calibrated value, refine the result before the binary
//!   arithmetic coder.
//! * **Two-speed coding.** Bytes deep inside a verified match (>= FAST_LEN)
//!   are coded by a tiny match-confidence SSE instead of the full model —
//!   deterministically, since both sides track the match length — making
//!   redundant regions nearly free in time as well as bits.
//! * **Runtime-SIMD mixer.** The first-layer dot products and weight updates
//!   run on AVX2 when available, with a bit-identical scalar fallback, so
//!   the bitstream never depends on the CPU.
//! * **Two profiles, four memory tiers.** Turbo (levels 1-3) runs a 5-model
//!   prefix of the roster with two mixer views and two APMs; full runs
//!   everything, at one of four table-size tiers. Both are recorded in the
//!   payload header, so decoding never depends on the level mapping.

use std::sync::OnceLock;

/// Model hyper-parameter lookup. With the `tune` feature the value can be
/// overridden by an environment variable of the same name, so competing
/// variants can be measured from a single build; without it (every shipped
/// build) this is a compile-time constant fold to `default`, and the bitstream
/// therefore never depends on the environment.
#[inline]
fn tunable(_name: &str, default: i32) -> i32 {
    #[cfg(feature = "tune")]
    {
        if let Ok(v) = std::env::var(_name) {
            if let Ok(n) = v.parse::<i32>() {
                return n;
            }
        }
    }
    default
}

// Hashing multipliers (odd, good avalanche).
const PR1: u32 = 0x9E37_79B1;
const PR2: u32 = 0x85EB_CA77;

// Table-size exponents are derived deterministically from the input byte
// count, which both encoder and decoder know, so the two sides always agree.
const HBITS_MAX: u32 = 22;
const HBITS_MIN: u32 = 14;

const MATCH_MIN: usize = 4; // short-hash suffix length that seeds a new match
const MATCH_MIN_LONG: usize = 8; // long-hash suffix length (tried first)
const MATCH_EMPTY: u32 = u32::MAX;

// Two-speed coding: once a verified match reaches this length, whole bytes
// are coded by a tiny adaptive match-confidence model instead of the full
// 26-model mixer — the byte is almost certainly the match continuation, so
// the heavy machinery would only sharpen an already near-certain prediction.
// The switch depends only on match_len, which encoder and decoder track in
// lockstep, so it is perfectly deterministic.
const FAST_LEN: u32 = 128;

/// Unclamped size target for `n` input bytes: a table a few times larger
/// than the input.
fn raw_bits(n: usize) -> u32 {
    (usize::BITS - n.max(1).leading_zeros()) + 2
}

/// Pick a power-of-two table exponent appropriate for `n` input bytes.
fn table_bits(n: usize) -> u32 {
    raw_bits(n).clamp(HBITS_MIN, HBITS_MAX)
}

/// Bucket-count exponent for bit-history model `k`. The `big` profile
/// (levels >= 7) grows the hashed-context tables 8x: on a large text segment
/// the population of distinct order-4..7 and word contexts vastly exceeds
/// the standard tables, and evictions were costing more ratio than any other
/// single factor. Sparse/stride contexts are low-cardinality, so they stay
/// capped regardless.
fn model_bits(k: usize, n: usize, mem: u8) -> u32 {
    // `raw_bits` is deliberately unclamped here: a 100 MB segment needs
    // 2^23+-bucket tables (128+ MiB per hashed model), and the standard
    // HBITS_MAX clamp was silently capping the big profile at 2^22 — the
    // second half of a big segment then thrashed the tables and levels 8-9
    // compressed *worse* than level 7. MEM_PLUS doubles every cap again so
    // a single 100 MB segment carries the same per-byte table pressure as
    // two 50 MB segments, while keeping the longer match window.
    let plus = (mem >= MEM_PLUS) as u32 + (mem >= MEM_HUGE) as u32;
    let bits = if mem >= MEM_BIG {
        raw_bits(n).clamp(11, 23 + plus)
    } else {
        // The standard profile is sized to stay small, so the low-cardinality
        // kinds are clamped harder than their populations alone would need.
        let h = raw_bits(n)
            .clamp(HBITS_MIN, HBITS_MAX)
            .saturating_sub(3)
            .clamp(11, 19);
        match MODEL_KIND[k] {
            Kind::Hash => h,
            Kind::Sparse | Kind::Stride => h.min(16),
            Kind::Ind => h.min(18),
        }
    };
    bits.min(MODEL_CAP[k])
}

/// Per-model ceiling on bucket-count exponent: how large a table the model's
/// context population can actually fill.
///
/// A context does not occupy one bucket. The high nibble takes one, and the
/// low nibble's hash is salted by the four bits already coded, so a context
/// that is visited with every possible high nibble touches 17 buckets. The
/// budget for a model with `c` distinct contexts is therefore `c * 17`, and
/// these caps are that figure rounded up with a factor of two of headroom for
/// hash collisions.
///
/// Bounding the genuinely small models — the order-2 context has 65,536
/// values, a stride lane 65,536, the nesting context about 16,000 — is what
/// pays for the much larger tables the high-order, word and indirect models
/// get in the top profiles. Those are the ones that actually thrash: on a
/// 1 GB segment they see hundreds of millions of distinct contexts.
const MODEL_CAP: [u32; NBH] = [
    21, 24, 31, 31, // orders 2-5
    31,             // word
    31, 31,         // orders 6-7
    31,             // word pair
    14, 14,         // sparse: one byte of context each
    21, 21, 21, 21, // sparse: two bytes of context each
    21, 21, 21, 21, // strides: two samples of one lane
    21, 24, 24,     // indirect order-2, -3, -4
    31, 31,         // orders 8, 10
    31, 31,         // orders 12, 16
    24,             // case-folded order-3
    24,             // enclosing element
    19,             // bracket nesting: delimiter x depth x previous byte
    24,             // line shape
    31,             // word trigram
];

/// Roughly how many bytes one [`Predictor`] over an `n`-byte segment will
/// allocate. Used to decide how many segments may be worked on at once: the
/// top profiles build multi-gigabyte tables, and running one per core would
/// exhaust any machine. Only the large allocations are counted, which is every
/// one that matters at these sizes.
pub fn model_bytes(n: usize, turbo: bool, mem: u8) -> usize {
    let nbh = if turbo { NBH_TURBO } else { NBH };
    let mem = if turbo { MEM_STD } else { mem };
    let mut total: usize = (0..nbh)
        .map(|k| BUCKET << model_bits(k, n, mem))
        .sum();

    // Match tables: two u32 tables, sized like Predictor::new does it.
    let mbits = if mem >= MEM_BIG {
        raw_bits(n).clamp(
            HBITS_MIN,
            24 + (mem >= MEM_PLUS) as u32 + (mem >= MEM_HUGE) as u32,
        )
    } else {
        table_bits(n)
    };
    total += 2 * 4 * (1usize << mbits);

    // Indirect byte tables, the mixer weight banks and the order-6 SSE stage.
    let ind_bits: u32 = if mem >= MEM_BIG { 22 } else { 20 };
    total += (1usize << 16) + 3 * (1usize << ind_bits);
    let mix_rows = mix_rows_for(n, turbo, mem);
    total += 3 * mix_rows * NINP * 4 + mix_rows * 33 * 2;

    // The segment's own byte history.
    total + n
}

// Memory profiles (recorded in the payload so decode always agrees).
pub const MEM_STD: u8 = 0;
pub const MEM_BIG: u8 = 1; // levels 7+: up to 2^23-bucket hash tables
pub const MEM_PLUS: u8 = 2; // level 8: up to 2^24 buckets, 2^25 match slots
pub const MEM_HUGE: u8 = 3; // level 9: up to 2^25 buckets, 2^26 match slots

const RATE_FAST: i32 = 3;

// Count-adaptive learning rate, as a 16-bit fraction:
// `RATE16[cnt] == round(2^16 / (cnt + 2))`. A freshly seen context (cnt == 0)
// moves halfway toward each observed bit; as evidence accumulates the step
// shrinks, so the estimate converges to the true stationary probability
// instead of jittering at a fixed rate. The count saturates at CNT_MAX, which
// floors the rate so the model can still track slow drift.
const CNT_MAX: usize = 255;
const RATE16: [u16; CNT_MAX + 1] = {
    let mut t = [0u16; CNT_MAX + 1];
    let mut i = 0;
    while i <= CNT_MAX {
        t[i] = ((1u32 << 16) / (i as u32 + 2)) as u16;
        i += 1;
    }
    t
};

// ---------------------------------------------------------------------------
// Bit-history states
// ---------------------------------------------------------------------------
// A state is one byte: high nibble = capped count of 0s, low nibble = capped
// count of 1s. On update the observed bit's count saturates upward while a
// large opposite count is *discounted* — so 0x0F ("fifteen 1s") and 0x4F
// ("recent 0s among many 1s") are distinct states even though plain counters
// would smear them together. The state map *learns* what each state predicts,
// so the exact discount schedule only shapes the state space, not the
// probabilities themselves.

/// Advance a packed (n0, n1) state by one observed bit.
#[inline]
fn state_next(s: u8, bit: i32) -> u8 {
    let mut n0 = s >> 4;
    let mut n1 = s & 15;
    if bit != 0 {
        n1 = (n1 + 1).min(15);
        if n0 > 3 {
            n0 = (n0 >> 1) + 1;
        }
    } else {
        n0 = (n0 + 1).min(15);
        if n1 > 3 {
            n1 = (n1 >> 1) + 1;
        }
    }
    (n0 << 4) | n1
}

/// Closed-form stretched estimate per state: Krichevsky–Trofimov
/// `p = (2*n1 + 1) / (2*n0 + 2*n1 + 2)`, stretched into the logistic domain.
fn st_direct_tbl() -> &'static [i16; 256] {
    static T: OnceLock<[i16; 256]> = OnceLock::new();
    T.get_or_init(|| {
        let mut t = [0i16; 256];
        for (s, slot) in t.iter_mut().enumerate() {
            let n0 = (s >> 4) as f64;
            let n1 = (s & 15) as f64;
            let p = (2.0 * n1 + 1.0) / (2.0 * n0 + 2.0 * n1 + 2.0);
            let p12 = (p * 4096.0).round().clamp(1.0, 4095.0) as i32;
            *slot = stretch(p12) as i16;
        }
        t
    })
}

/// A state-map entry: learned 16-bit `P(bit==1)` for one bit-history state,
/// adapted at the count-adaptive RATE16 schedule.
#[derive(Clone, Copy)]
struct SmEntry {
    p: u16,
    cnt: u16,
}

impl SmEntry {
    #[inline]
    fn update(&mut self, bit: i32) {
        let target = bit << 16;
        let p = self.p as i32;
        let rate = RATE16[self.cnt as usize] as i32;
        self.p = (p + (((target - p) * rate) >> 16)) as u16;
        if (self.cnt as usize) < CNT_MAX {
            self.cnt += 1;
        }
    }
}

/// Number of state-map planes per bit-history model: one per depth within the
/// nibble (0..3). The same packed count state means different things at
/// different levels of the nibble tree — the root node of a bucket sees every
/// visit to the context while a depth-3 node only sees the sliver of visits
/// that shared the three preceding bits — so giving each depth its own learned
/// map lets both calibrate independently.
const SM_DEPTHS: usize = 4;
const SM_SIZE: usize = 256 * SM_DEPTHS;

/// A fresh state map, with every entry seeded from its state's closed-form
/// estimate rather than 0.5 — a brand-new context predicts sensibly from its
/// very first visit, and the count-adaptive rate then refines from there.
fn sm_init() -> Vec<SmEntry> {
    let mut t = vec![SmEntry { p: 32768, cnt: 0 }; SM_SIZE];
    for (i, e) in t.iter_mut().enumerate() {
        let s = i & 255;
        let n0 = (s >> 4) as f64;
        let n1 = (s & 15) as f64;
        let p = (2.0 * n1 + 1.0) / (2.0 * n0 + 2.0 * n1 + 2.0);
        e.p = (p * 65536.0).round().clamp(1.0, 65535.0) as u16;
    }
    t
}

// ---------------------------------------------------------------------------
// Nibble-bucketed hash table of bit-history states
// ---------------------------------------------------------------------------
// Bucket layout (16 bytes): [checksum | 15 states]. The 15 states cover the
// complete binary subtree of one nibble (1 root + 2 + 4 + 8), indexed by the
// nibble-local path register. One find() per nibble serves four bits.

const BUCKET: usize = 16;

struct BhTable {
    t: Vec<u8>,
    mask: u32, // bucket-index mask
    ways: u32, // candidate buckets probed per lookup (2 or 4)
}

impl BhTable {
    fn new(bucket_bits: u32, ways: u32) -> Self {
        Self {
            t: vec![0u8; BUCKET << bucket_bits],
            mask: (1u32 << bucket_bits) - 1,
            // Candidates are the bucket index with its low bits varied, so an
            // n-way set is one aligned run of n buckets — at most two cache
            // lines, and usually one.
            ways,
        }
    }

    /// Hint both candidate buckets for `h` into cache (they are usually the
    /// same 64-byte line). Called for every model *before* the find() pass so
    /// the memory latencies overlap.
    #[inline]
    fn prefetch(&self, h: u32) {
        // A 2- or 4-way set is 32 or 64 aligned bytes, so it never straddles a
        // cache line and the second hint is free; at 8 ways it is the second
        // line.
        let i0 = ((h & self.mask & !(self.ways - 1)) as usize) * BUCKET;
        prefetch_ptr(unsafe { self.t.as_ptr().add(i0) });
        prefetch_ptr(unsafe { self.t.as_ptr().add(i0 + (self.ways as usize - 1) * BUCKET) });
    }

    /// Find (or allocate) the bucket for hash `h`; returns the byte offset of
    /// its 15-state slot array. Two candidate buckets are probed; on a double
    /// miss the bucket whose root state carries less evidence is recycled.
    #[inline]
    fn find(&mut self, h: u32) -> usize {
        let cs = ((h >> 24) as u8) | 1; // 0 marks "never used"
        let ways = self.ways as usize;
        let base = ((h & self.mask & !(self.ways - 1)) as usize) * BUCKET;
        // Hit?
        for w in 0..ways {
            let i = base + w * BUCKET;
            if self.t[i] == cs {
                return i + 1;
            }
        }
        // Miss: take a free bucket, else recycle the least established one —
        // the total observation count at the root slot says how much evidence
        // a bucket would lose.
        let mut victim = base;
        let mut victim_ev = u32::MAX;
        for w in 0..ways {
            let i = base + w * BUCKET;
            if self.t[i] == 0 {
                victim = i;
                break;
            }
            let e = self.t[i + 1];
            let ev = ((e >> 4) + (e & 15)) as u32;
            if ev < victim_ev {
                victim_ev = ev;
                victim = i;
            }
        }
        self.t[victim] = cs;
        self.t[victim + 1..victim + BUCKET].fill(0);
        victim + 1
    }
}

#[inline]
fn prefetch_ptr(p: *const u8) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        std::arch::x86_64::_mm_prefetch(p as *const i8, std::arch::x86_64::_MM_HINT_T0)
    };
    #[cfg(not(target_arch = "x86_64"))]
    let _ = p;
}

// ---------------------------------------------------------------------------
// Model roster
// ---------------------------------------------------------------------------

// Sparse "stride" models capture fixed-period structure in binary media:
// 16-bit / stereo audio (stride 2, 4), RGB / RGBA images (stride 3, 4), and
// many fixed-record game formats. Each predicts the current byte from the
// same lane of previous samples. The mixer learns to trust them on media and
// ignore them on text, so they are safe to always include.
const STRIDES: [usize; 4] = [2, 3, 4, 8];
const NSTRIDE: usize = STRIDES.len();

// Bit-history models: hashed orders 2..7, the current word, the
// previous-word/current-word pair, two sparse contexts (skip-gram and
// high-nibble), the four stride contexts, and two *indirect* contexts
// (keyed by the byte that followed the same context last time — strong on
// natural-language text, where "what came after this bigram before" is a
// better cue than the bigram alone). Sparse and stride contexts are
// low-cardinality, so their tables are capped smaller.
const NHASH: usize = 8;
const NSPARSE: usize = 6;
const NIND: usize = 3;
const NTEXT: usize = 5; // order-8, order-10, order-12, order-16, case-folded order-3
// Markup models. Wikipedia dumps — and XML, HTML, JSON, source code and config
// files generally — are *structured* text: what predicts the next byte is
// often not the preceding bytes but which element encloses them, how deep the
// bracket nesting is, and how far into the line we are. None of the order-n,
// word or sparse contexts can see any of that.
//   * enclosing element — the innermost open tag name, plus one byte of local
//     context, so `<title>` content and `<timestamp>` content stop sharing
//     statistics;
//   * nesting — the innermost unclosed bracket/quote delimiter and its depth,
//     which is what separates wiki `[[link]]`, `{{template}}` and plain prose;
//   * line shape — the column and the bytes at the start of the line, which
//     carry the indentation and list-marker conventions;
//   * word trigram — the two previous whole words plus the current word's
//     prefix, a genuine language model context that the word-pair model only
//     approximates.
const NMARKUP: usize = 4;
const NBH: usize = NHASH + NSPARSE + NSTRIDE + NIND + NTEXT + NMARKUP; // 30 models

// Per-model table kind, indexed like `bh_base`: how big a hash table the
// model's context population deserves.
#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Hash,   // hashed high-order / word contexts: unbounded population
    Sparse, // skip-grams: bounded by 2 bytes of context
    Stride, // fixed-lane media contexts: low cardinality
    Ind,    // indirect contexts: bounded by (byte, order-1/2)
}
const MODEL_KIND: [Kind; NBH] = [
    Kind::Hash, Kind::Hash, Kind::Hash, Kind::Hash, // orders 2-5
    Kind::Hash,                                     // word
    Kind::Hash, Kind::Hash,                         // orders 6-7
    Kind::Hash,                                     // word pair
    Kind::Sparse, Kind::Sparse, Kind::Sparse, Kind::Sparse, Kind::Sparse, Kind::Sparse,
    Kind::Stride, Kind::Stride, Kind::Stride, Kind::Stride,
    Kind::Ind, Kind::Ind, Kind::Ind,               // indirect order-2, -3, -4
    Kind::Hash, Kind::Hash,                         // orders 8, 10
    Kind::Hash, Kind::Hash,                         // orders 12, 16
    Kind::Hash,                                     // case-folded order-3
    Kind::Ind,                                      // enclosing element
    Kind::Sparse,                                   // bracket nesting
    Kind::Sparse,                                   // line shape
    Kind::Hash,                                     // word trigram
];
// The turbo profile (levels 1-3) runs only the first NBH_TURBO models
// (orders 2-5 + word), two mixer views and two APMs — a several-times-faster
// engine that still beats the classical tools on ratio.
const NBH_TURBO: usize = 5;

// First-layer mixer inputs:
//   order-0 + order-1 dual counters (2 each)
// + NBH bit-history models * 2 (state map + direct state estimate)
// + 1 match model
// + 1 bias
const NIN: usize = 4 + NBH * 2 + 2;
// Weight rows are padded to a multiple of 8 lanes for the SIMD mixer; the pad
// inputs are always zero, so they contribute nothing and learn nothing.
const NINP: usize = (NIN + 7) & !7;
const BH_IN: usize = 4; // first bit-history input index
const MATCH_IN: usize = BH_IN + NBH * 2;
const BIAS_IN: usize = NIN - 1;


// First-layer weight-set row counts. The `wc` view is selected by a hashed
// order-3 context rather than the single previous-previous byte: on text,
// which short run of bytes precedes the position is a far sharper cue for
// which models to trust than any single byte. More rows let the mixer keep
// distinct weight vectors for many more distinct local contexts; on big text
// each row still sees ample training traffic.
// Rows in each hash-selected first-layer weight bank (`wc` order-3, `we`
// order-6, `wg` word), and in the order-6 SSE stage. Each row is an
// independent weight vector, so more rows means the mixer can hold a distinct
// blend for more distinct local contexts instead of averaging them together —
// on 16 MiB of enwik8 going from 2^13 to 2^17 rows is worth 0.45%, still
// improving at the top. The banks are sized by memory profile and bounded by
// the segment length, since a row that is never selected only wastes memory.
const WC_ROWS: usize = 8192;
const WC_ROWS_BIG: usize = 1 << 16;
const WC_ROWS_PLUS: usize = 1 << 18;
const WC_ROWS_HUGE: usize = 1 << 19;

/// Rows in each hash-selected first-layer weight bank, for an `n`-byte segment.
fn mix_rows_for(n: usize, turbo: bool, mem: u8) -> usize {
    if turbo {
        return WC_ROWS;
    }
    let by_profile = match mem {
        MEM_STD => WC_ROWS,
        MEM_BIG => WC_ROWS_BIG,
        MEM_PLUS => WC_ROWS_PLUS,
        _ => WC_ROWS_HUGE,
    };
    // A row that is never selected only wastes memory, so never exceed what
    // the segment can plausibly train.
    let by_input = (n / 64).next_power_of_two().max(WC_ROWS);
    (tunable("CPGC_MIX_ROWS", by_profile.min(by_input) as i32) as usize).next_power_of_two()
}

// Second-layer mixer: combines the six first-layer outputs plus a bias.
const NMIX: usize = 7;
// Selected by (classes of the two previous bytes, min(match_len, 7), bit
// position): the combiner learns, e.g., to trust the match view less on low
// bits and counter views more there — with separate weights per character
// class (letter, space, digit, other), which on text is a sharper predictor of
// which views to trust than a single in-word flag. Two bytes of class rather
// than one distinguishes the start of a word from its interior, and a digit
// inside a number from one just after a letter.
const NMIX_CTX: usize = 1024;

/// Coarse character class of a byte, used to select second-layer mixer weights:
/// 0 = ASCII letter, 1 = space/tab/newline, 2 = ASCII digit, 3 = everything
/// else (punctuation, markup, high bytes).
#[inline]
fn char_class(b: u8) -> usize {
    if b.is_ascii_alphabetic() {
        0
    } else if b == b' ' || b == b'\n' || b == b'\t' || b == b'\r' {
        1
    } else if b.is_ascii_digit() {
        2
    } else {
        3
    }
}

// Mixer learning rate: how hard each coding error pushes the weights.
//
// The right rate falls as the segment gets longer, and it matters far more
// than its size suggests. A rate that is right for a 16 MiB segment is much
// too fast for a 128 MiB one — the weights chase noise instead of settling —
// and the cost is not subtle: on a single 128 MiB segment, rate 4 gives
// 25,258,023 bytes where rate 3 gives 23,925,575, a 5.3% difference. Getting
// this wrong is what made large segments look like a dead end; with the rate
// matched to the segment, one 128 MiB segment beats two 64 MiB ones by 2.7%.
//
// So the rate is a function of the segment length, which encoder and decoder
// both know before they start, and it is never stored.
const MIX_LR_TURBO: i32 = 5;
/// Mixer learning rate for a segment of `n` bytes.
fn mix_lr_for(n: usize, turbo: bool) -> i32 {
    if turbo {
        // Turbo only ever runs 1-4 MiB segments, well inside the fast regime.
        return MIX_LR_TURBO;
    }
    // Only two rates are warranted by measurement. Going lower still is not a
    // safe extrapolation — it is actively wrong: on a 256 MB segment rate 2
    // gives 48,252,748 bytes against rate 3's 45,980,493, giving back almost
    // everything the larger window won.
    if n <= 64 << 20 {
        4
    } else {
        3
    }
}
// Upper bound over every profile, used only to size the SIMD-equivalence
// test's range.
#[allow(dead_code)] // referenced only from the (cfg(test)) SIMD-equivalence test
const MIX_LR: i32 = MIX_LR_TURBO;
// First-layer weight clamp. ±2^19 at 16 fractional bits (gain ±8) keeps every
// weight-input product inside i32, which the AVX2 mixer path relies on.
const W_CLAMP: i32 = (1 << 19) - 1;

// ---------------------------------------------------------------------------
// Logistic transfer tables (shared, built once).
// ---------------------------------------------------------------------------

fn squash_tbl() -> &'static [i16; 4096] {
    static T: OnceLock<[i16; 4096]> = OnceLock::new();
    T.get_or_init(|| {
        let mut t = [0i16; 4096];
        for (i, slot) in t.iter_mut().enumerate() {
            let d = (i as i32 - 2048) as f64;
            let p = 4096.0 / (1.0 + (-d / 256.0).exp());
            *slot = p.round().clamp(1.0, 4095.0) as i16;
        }
        t
    })
}

fn stretch_tbl() -> &'static [i16; 4096] {
    static T: OnceLock<[i16; 4096]> = OnceLock::new();
    T.get_or_init(|| {
        let mut t = [0i16; 4096];
        for (p, slot) in t.iter_mut().enumerate() {
            let pc = (p as f64).clamp(1.0, 4095.0);
            let d = 256.0 * (pc / (4096.0 - pc)).ln();
            *slot = d.round().clamp(-2047.0, 2047.0) as i16;
        }
        t
    })
}

#[inline]
fn squash(d: i32) -> i32 {
    let d = d.clamp(-2047, 2047);
    squash_tbl()[(d + 2048) as usize] as i32
}

#[inline]
fn stretch(p: i32) -> i32 {
    stretch_tbl()[p.clamp(0, 4095) as usize] as i32
}

/// A dual-rate bit counter: a fast (fixed-rate, reactive) and a slow
/// (count-adaptive, converging) 16-bit `P(bit==1)` estimate, plus the visit
/// count that drives the slow estimate's shrinking learning rate. Used for
/// the small direct-indexed order-0/1 tables, which are collision-free and
/// cache-resident, so the richer 6-byte slot is affordable there.
#[derive(Clone, Copy)]
struct DualCounter {
    fast: u16,
    slow: u16,
    cnt: u16,
}

impl DualCounter {
    const INIT: DualCounter = DualCounter {
        fast: 32768,
        slow: 32768,
        cnt: 0,
    };

    #[inline]
    fn update(&mut self, bit: i32) {
        let target = bit << 16;
        let f = self.fast as i32;
        self.fast = (f + ((target - f) >> RATE_FAST)) as u16;
        let s = self.slow as i32;
        let rate = RATE16[self.cnt as usize] as i32;
        self.slow = (s + (((target - s) * rate) >> 16)) as u16;
        if (self.cnt as usize) < CNT_MAX {
            self.cnt += 1;
        }
    }
}

// ---------------------------------------------------------------------------
// SSE / APM: refines a probability using a small context via interpolation
// over 33 nodes laid out evenly in the stretch domain.
// ---------------------------------------------------------------------------

struct Apm {
    t: Vec<u16>,
    base: usize,
    w: i32,
}

impl Apm {
    fn new(n: usize) -> Self {
        let mut t = vec![0u16; n * 33];
        for c in 0..n {
            for j in 0..33 {
                let p = squash((j as i32 - 16) * 128);
                t[c * 33 + j] = (p << 4) as u16;
            }
        }
        Self { t, base: 0, w: 0 }
    }

    #[inline]
    fn refine(&mut self, pr: i32, cxt: usize) -> i32 {
        let s = (stretch(pr) + 2048).clamp(0, 4095);
        let j = (s >> 7) as usize;
        let w = s & 127;
        self.base = cxt * 33 + j;
        self.w = w;
        let lo = self.t[self.base] as i32;
        let hi = self.t[self.base + 1] as i32;
        let p16 = (lo * (128 - w) + hi * w) >> 7;
        (p16 >> 4).clamp(1, 4095)
    }

    /// Update *both* interpolation endpoints, each in proportion to the
    /// weight it contributed to the prediction — the same total learning
    /// rate as a single-node update, but the map stays smooth instead of
    /// developing staircase artifacts at node boundaries.
    #[inline]
    fn update(&mut self, bit: i32) {
        let target = bit << 16;
        let lo = self.t[self.base] as i32;
        let hi = self.t[self.base + 1] as i32;
        self.t[self.base] = (lo + (((target - lo) * (128 - self.w)) >> 13)) as u16;
        self.t[self.base + 1] = (hi + (((target - hi) * self.w) >> 13)) as u16;
    }
}

// ---------------------------------------------------------------------------
// Predictor
// ---------------------------------------------------------------------------

pub struct Predictor {
    // Direct-indexed dual counters (small, collision-free).
    t0: Vec<DualCounter>, // order-0: partial byte (256)
    t1: Vec<DualCounter>, // order-1: prev1<<8 | c0 (65536)
    idx0: usize,
    idx1: usize,

    // Rolling byte history (hist[0] = most recent).
    hist: [u8; 16],
    word_hash: u32,
    last_word: u32,  // hash of the most recently *finished* word
    prev_word: u32,  // the one before that (word-trigram context)

    // Markup state, all derived from committed bytes only.
    tag_hash: u32,       // innermost open element name
    tag_stack: [u32; 8], // enclosing element names, tag_depth deep
    tag_depth: usize,
    tag_name: u32,   // element name currently being scanned
    tag_scan: u8,    // 0 = outside a tag, 1 = reading a name, 2 = past the name
    tag_close: bool, // the tag being scanned is a closing tag
    nest: [u8; 16],  // stack of unclosed bracket/quote delimiters
    nest_depth: usize,
    col: u32,        // bytes since the last newline
    line_head: u32,  // hash of the first few bytes of the current line

    // Indirect context state: per-context "byte that followed last time".
    // ind2 is direct-indexed by the order-2 context (collision-free); ind3
    // and ind4 are indexed by hashed order-3 / order-4 contexts. A collision
    // only yields a noisy context input — never an incorrect decode.
    ind2: Vec<u8>,
    ind3: Vec<u8>,
    ind4: Vec<u8>,
    ind3_mask: u32,
    ind4_mask: u32,
    ind2_idx: usize, // slot to write the *next* committed byte into
    ind3_idx: usize,
    ind4_idx: usize,
    indt: Vec<u8>, // per-element: the byte that last followed this element context
    indt_mask: u32,
    indt_idx: usize,

    // Bit-history models: nibble-bucketed state tables + per-model state maps.
    bh: Vec<BhTable>,            // NBH tables
    bh_sm: Vec<Vec<SmEntry>>,    // NBH state maps, SM_DEPTHS planes each
    bh_base: [u32; NBH],         // per-byte context hashes
    bh_off: [usize; NBH],        // resolved bucket slot-array offsets
    bh_state: [u8; NBH],         // states read by predict(), for update()
    nib_path: u32,               // nibble-local path register (1..15)
    pending_hs: [u32; NBH],      // low-nibble hashes, prefetched a bit early
    nbh: usize,                  // active model count (NBH_TURBO or NBH)
    turbo: bool,                 // reduced mixer/SSE roster for low levels
    mix_lr: i32,                 // mixer learning rate (per profile)
    // Ablation switches. Always 1 in shipped builds (see `tunable`); the
    // `tune` feature lets a measurement run turn a single change off.
    mix2_shift: usize, // 2 when the second layer sees both previous classes
    mix2_mask: usize,
    sm_planes: usize, // 1 = one shared state map, SM_DEPTHS = one per depth

    // Partial byte: starts at 1, accumulates coded bits.
    c0: u32,

    // Match model. Two suffix-hash tables: the long (8-byte) hash is tried
    // first when seeding so matches start from more reliable anchors.
    buf: Vec<u8>,
    match_table: Vec<u32>,      // 4-byte suffix hash
    match_table_long: Vec<u32>, // 8-byte suffix hash
    match_mask: u32,
    match_ptr: usize,
    match_len: u32,
    match_byte: i32, // predicted next byte, or -1 if no active match

    // Two-speed coding state. fast_mode is fixed per byte (at next_byte);
    // fast_p is a tiny SSE keyed by (match-length bucket, bit position,
    // predicted bit); fast_state remembers which sub-model predicted the
    // current bit (1 = match SSE, 2 = order-0/1 fallback after a break).
    fast_len: u32,
    fast_mode: bool,
    fast_state: u8,
    fast_idx: usize,
    fast_p: [u16; 256],

    // First-layer mixer: five context-selected weight sets.
    wa: Vec<i32>, // [256][NINP] selected by previous byte
    wb: Vec<i32>, // [64][NINP]  selected by match-length bucket
    wc: Vec<i32>, // [mix_rows][NINP] selected by a hashed order-3 context
    wd: Vec<i32>, // [256][NINP] selected by the partial byte (c0)
    we: Vec<i32>, // [mix_rows][NINP] selected by a hashed order-6 context
    wg: Vec<i32>, // [mix_rows][NINP] selected by the current word hash
    mix_mask: usize, // mix_rows - 1
    tx: [i32; NINP],
    use_avx2: bool, // AVX2 detected at runtime (paths are bit-identical)
    ctx_a: usize,
    ctx_b: usize,
    ctx_c: usize,
    ctx_d: usize,
    ctx_e: usize,
    ctx_g: usize,

    // Second-layer mixer: combines the six first-layer outputs.
    wf: Vec<i32>, // [NMIX_CTX][NMIX]
    mi: [i32; NMIX],
    ctx_f: usize,
    pr: i32,

    // SSE chain.
    apm0: Apm,
    apm1: Apm,
    apm2: Apm,
    apm3: Apm,
    final_pr: i32,
}

impl Predictor {
    /// `turbo` selects the reduced low-level profile; `mem` selects the
    /// memory profile (MEM_STD / MEM_BIG / MEM_PLUS). Both change the
    /// bitstream, so the codec records them in the payload header.
    pub fn new(n: usize, turbo: bool, mem: u8) -> Self {
        let _ = squash(0);
        let _ = stretch(2048);
        let _ = st_direct_tbl();

        let nbh = if turbo {
            NBH_TURBO
        } else {
            (tunable("CPGC_NBH", NBH as i32) as usize).clamp(NBH_TURBO, NBH)
        };

        // Rows in each context-selected first-layer weight bank. Turbo never
        // uses the hashed views, so it keeps the smallest bank.
        let mix_rows = mix_rows_for(n, turbo, mem);

        // Set associativity of the bit-history tables. More ways means fewer
        // contexts evicted by an unlucky hash, which matters most when the
        // context population dwarfs the table.
        let ways = (tunable("CPGC_WAYS", if mem >= MEM_PLUS { 4 } else { 2 }) as u32)
            .next_power_of_two()
            .clamp(2, 8);

        // The match tables store one u32 per slot; the big profiles grow
        // them so long-range matches on a 100 MB+ segment survive (raw_bits,
        // not table_bits: the standard clamp must not cap the big profiles).
        let mbits = if mem >= MEM_BIG {
            raw_bits(n).clamp(
                HBITS_MIN,
                24 + (mem >= MEM_PLUS) as u32 + (mem >= MEM_HUGE) as u32,
            )
        } else {
            table_bits(n)
        };
        let msize = 1usize << mbits;

        let ind3_bits: u32 = if mem >= MEM_BIG { 22 } else { 20 };
        let ind4_bits: u32 = if mem >= MEM_BIG { 22 } else { 20 };

        // Initialise the second-layer weights so the mixer starts out close to
        // an average of its active view inputs (two in turbo, five in full);
        // gradient descent refines from there.
        let views = if turbo { 2 } else { 6 };
        let avg_w = ((1i64 << 16) / views) as i32;
        let mut wf = vec![0i32; NMIX_CTX * NMIX];
        for c in 0..NMIX_CTX {
            if turbo {
                wf[c * NMIX] = avg_w; // sa
                wf[c * NMIX + 3] = avg_w; // sd
            } else {
                for i in 0..NMIX - 1 {
                    wf[c * NMIX + i] = avg_w;
                }
            }
            // the bias weight (last slot) stays 0
        }

        Self {
            t0: vec![DualCounter::INIT; 256],
            t1: vec![DualCounter::INIT; 1 << 16],
            idx0: 0,
            idx1: 0,
            hist: [0; 16],
            word_hash: 0,
            last_word: 0,
            prev_word: 0,
            tag_hash: 0,
            tag_stack: [0; 8],
            tag_depth: 0,
            tag_name: 0,
            tag_scan: 0,
            tag_close: false,
            nest: [0; 16],
            nest_depth: 0,
            col: 0,
            line_head: 0,
            indt: vec![0u8; 1 << ind3_bits],
            indt_mask: (1u32 << ind3_bits) - 1,
            indt_idx: 0,
            ind2: vec![0u8; 1 << 16],
            ind3: vec![0u8; 1 << ind3_bits],
            ind4: vec![0u8; 1 << ind4_bits],
            ind3_mask: (1u32 << ind3_bits) - 1,
            ind4_mask: (1u32 << ind4_bits) - 1,
            ind2_idx: 0,
            ind3_idx: 0,
            ind4_idx: 0,
            bh: (0..nbh)
                .map(|k| BhTable::new(model_bits(k, n, mem), ways))
                .collect(),
            bh_sm: vec![sm_init(); nbh],
            nbh,
            turbo,
            mix_lr: tunable("CPGC_MIX_LR", mix_lr_for(n, turbo)),
            mix2_shift: if tunable("CPGC_MIX2CTX", 1) != 0 { 2 } else { 0 },
            mix2_mask: if tunable("CPGC_MIX2CTX", 1) != 0 { 3 } else { 0 },
            sm_planes: if tunable("CPGC_SM_DEPTH", 1) != 0 { SM_DEPTHS } else { 1 },
            bh_base: [0; NBH],
            bh_off: [1; NBH],
            bh_state: [0; NBH],
            nib_path: 1,
            pending_hs: [0; NBH],
            c0: 1,
            buf: Vec::with_capacity(n),
            match_table: vec![MATCH_EMPTY; msize],
            match_table_long: vec![MATCH_EMPTY; msize],
            match_mask: (msize as u32) - 1,
            match_ptr: 0,
            match_len: 0,
            match_byte: -1,
            fast_len: tunable("CPGC_FAST_LEN", FAST_LEN as i32).max(1) as u32,
            fast_mode: false,
            fast_state: 0,
            fast_idx: 0,
            fast_p: {
                // Seed: when the match predicts 1 expect ~0.95, else ~0.05;
                // the per-bucket entries adapt from there.
                let mut t = [0u16; 256];
                let mut i = 0;
                while i < 256 {
                    t[i] = if i & 1 == 1 { 3900 << 4 } else { 196 << 4 };
                    i += 1;
                }
                t
            },
            wa: vec![0i32; 256 * NINP],
            wb: vec![0i32; 64 * NINP],
            wc: vec![0i32; mix_rows * NINP],
            wd: vec![0i32; 256 * NINP],
            we: vec![0i32; mix_rows * NINP],
            wg: vec![0i32; mix_rows * NINP],
            mix_mask: mix_rows - 1,
            tx: [0; NINP],
            #[cfg(target_arch = "x86_64")]
            use_avx2: std::arch::is_x86_feature_detected!("avx2"),
            #[cfg(not(target_arch = "x86_64"))]
            use_avx2: false,
            ctx_a: 0,
            ctx_b: 0,
            ctx_c: 0,
            ctx_d: 0,
            ctx_e: 0,
            ctx_g: 0,
            wf,
            mi: [0; NMIX],
            ctx_f: 0,
            pr: 2048,
            apm0: Apm::new(256),
            apm1: Apm::new(1 << 16),
            apm2: Apm::new(16384),
            apm3: Apm::new(mix_rows),
            final_pr: 2048,
        }
    }

    /// Hashes for the low nibble's buckets, given the four high bits.
    #[inline]
    fn nib1_hashes(&self, nib0: u32) -> [u32; NBH] {
        let salt = (nib0 + 17).wrapping_mul(PR2);
        let mut hs = [0u32; NBH];
        for k in 0..self.nbh {
            let mut h = self.bh_base[k] ^ salt;
            h ^= h >> 15;
            hs[k] = h;
        }
        hs
    }

    /// Locate every model's bucket for the nibble that starts now. The
    /// candidate cache lines were already prefetched when the hashes first
    /// became known (end of `next_byte` for the high nibble, end of the 4th
    /// bit's `update` for the low one), so these find()s mostly hit lines
    /// that are already in flight.
    #[inline]
    fn resolve_buckets(&mut self, hs: &[u32; NBH]) {
        for k in 0..self.nbh {
            self.bh_off[k] = self.bh[k].find(hs[k]);
        }
        self.nib_path = 1;
    }

    /// Fast-path prediction inside a long verified match: a 256-entry SSE
    /// keyed by (match-length bucket, bit position, predicted bit), with an
    /// order-0/1 fallback if the match is contradicted mid-byte.
    #[inline]
    fn fast_predict(&mut self) -> i32 {
        let c0 = self.c0;
        let bits_seen = 31 - c0.leading_zeros(); // 0..7
        let mp = self.match_byte as u32; // fast_mode guarantees match_byte >= 0
        let coded = c0 - (1 << bits_seen);
        if coded == mp >> (8 - bits_seen) {
            let predicted_bit = ((mp >> (7 - bits_seen)) & 1) as usize;
            let bucket = (31 - self.match_len.leading_zeros()).min(15) as usize;
            let idx = (bucket << 4) | ((bits_seen as usize) << 1) | predicted_bit;
            self.fast_idx = idx;
            self.fast_state = 1;
            self.final_pr = ((self.fast_p[idx] >> 4) as i32).clamp(1, 4095);
        } else {
            // Match broke mid-byte: finish the byte on the order-0/1 counters.
            self.idx0 = c0 as usize;
            self.idx1 = ((self.hist[0] as usize) << 8) | (c0 as usize & 0xff);
            let p0 = self.t0[self.idx0].slow as i32;
            let p1 = self.t1[self.idx1].slow as i32;
            self.fast_state = 2;
            self.final_pr = (((p1 * 3 + p0) >> 2) >> 4).clamp(1, 4095);
        }
        self.final_pr
    }

    #[inline]
    pub fn predict(&mut self) -> i32 {
        if self.fast_mode {
            return self.fast_predict();
        }
        let c0 = self.c0;

        // Nibble boundary: re-anchor every bit-history model.
        if c0 == 1 {
            let hs = self.bh_base;
            self.resolve_buckets(&hs);
        } else if c0 >> 4 == 1 {
            let hs = self.pending_hs;
            self.resolve_buckets(&hs);
        }

        // --- order-0/1 dual counters: fast+slow stretched estimates ------
        self.idx0 = c0 as usize;
        let c = self.t0[self.idx0];
        self.tx[0] = stretch((c.fast >> 4) as i32);
        self.tx[1] = stretch((c.slow >> 4) as i32);

        self.idx1 = (((self.hist[0] as usize) << 8) | (c0 as usize & 0xff)) & 0xffff;
        let c = self.t1[self.idx1];
        self.tx[2] = stretch((c.fast >> 4) as i32);
        self.tx[3] = stretch((c.slow >> 4) as i32);

        // --- bit-history models: state map + direct state estimate -------
        let sidx = (self.nib_path - 1) as usize;
        let smp = self.sm_plane();
        let st_direct = st_direct_tbl();
        for k in 0..self.nbh {
            let s = self.bh[k].t[self.bh_off[k] + sidx];
            self.bh_state[k] = s;
            let e = self.bh_sm[k][smp + s as usize];
            self.tx[BH_IN + k * 2] = stretch((e.p >> 4) as i32);
            self.tx[BH_IN + k * 2 + 1] = st_direct[s as usize] as i32;
        }

        // --- match model -------------------------------------------------
        self.tx[MATCH_IN] = self.match_prediction(c0);

        // --- bias --------------------------------------------------------
        self.tx[BIAS_IN] = 256;

        // --- first-layer mixing: context-selected weight sets -------------
        // (turbo runs only the prev-byte and partial-byte views; ctx_c and
        // ctx_e are hashed order-3 / order-6 selections, computed once per
        // byte in next_byte)
        self.ctx_a = self.hist[0] as usize;
        self.ctx_b = (self.match_len.min(63)) as usize;
        self.ctx_d = (c0 & 0xff) as usize;
        // Each view's output is clamped to ±2^15. The bound is far outside
        // the useful stretch range (squash saturates near ±2047) so it never
        // costs prediction sharpness, but it keeps the second-layer update
        // products comfortably inside i64 *and* stops runaway feedback: with
        // fully unbounded dots the combiner destabilised after ~12 MiB of a
        // single segment and the model collapsed to ~4.4 bpb for the rest.
        const S_CLAMP: i32 = 1 << 15;
        let sa = self.dot(&self.wa, self.ctx_a).clamp(-S_CLAMP, S_CLAMP);
        let sd = self.dot(&self.wd, self.ctx_d).clamp(-S_CLAMP, S_CLAMP);
        let (sb, sc, se, sg) = if self.turbo {
            (0, 0, 0, 0)
        } else {
            (
                self.dot(&self.wb, self.ctx_b).clamp(-S_CLAMP, S_CLAMP),
                self.dot(&self.wc, self.ctx_c).clamp(-S_CLAMP, S_CLAMP),
                self.dot(&self.we, self.ctx_e).clamp(-S_CLAMP, S_CLAMP),
                self.dot(&self.wg, self.ctx_g).clamp(-S_CLAMP, S_CLAMP),
            )
        };

        // --- second-layer mixing: a small learned combiner ---------------
        self.mi = [sa, sb, sc, sd, se, sg, 256];
        let bits_seen = (31 - c0.leading_zeros()) as usize; // 0..7
        let cls = char_class(self.hist[0]) << self.mix2_shift
            | (char_class(self.hist[1]) & self.mix2_mask);
        self.ctx_f = cls << 6 | (self.match_len.min(7) as usize) << 3 | bits_seen;
        let mixed = self.dot2(self.ctx_f);
        self.pr = squash(mixed);

        // --- SSE refinement (turbo keeps apm0 + apm2 only) ----------------
        let p0 = self.apm0.refine(self.pr, self.ctx_d);
        let mut p = (self.pr * 3 + p0) >> 2;
        if !self.turbo {
            let o2 = ((self.hist[0] as usize) << 8) | self.hist[1] as usize;
            let p1 = self.apm1.refine(p, o2);
            p = (p * 3 + p1) >> 2;
        }
        let p2 = self.apm2.refine(p, (self.bh_base[1] & 0x3fff) as usize);
        p = (p * 3 + p2) >> 2;
        if !self.turbo {
            // Keyed by the order-6 context so this stage calibrates on a
            // longer context than apm2's order-3 hash, rather than duplicating
            // it.
            let p3 = self.apm3.refine(p, self.ctx_e);
            p = (p * 3 + p3) >> 2;
        }
        self.final_pr = p.clamp(1, 4095);
        self.final_pr
    }

    #[inline]
    fn dot(&self, w: &[i32], ctx: usize) -> i32 {
        let base = ctx * NINP;
        let row = &w[base..base + NINP];
        #[cfg(target_arch = "x86_64")]
        if self.use_avx2 {
            // SAFETY: only taken when AVX2 was detected at runtime.
            return unsafe { dot_avx2(row, &self.tx) };
        }
        dot_scalar(row, &self.tx)
    }

    #[inline]
    fn dot2(&self, ctx: usize) -> i32 {
        let base = ctx * NMIX;
        let mut acc = 0i64;
        for i in 0..NMIX {
            acc += (self.wf[base + i] as i64) * (self.mi[i] as i64);
        }
        (acc >> 16) as i32
    }

    /// Which state-map plane the current nibble depth selects.
    #[inline]
    fn sm_plane(&self) -> usize {
        if self.sm_planes == 1 {
            0
        } else {
            (31 - self.nib_path.leading_zeros()) as usize * 256
        }
    }

    /// Stretched prediction from the match model for the current partial byte.
    /// Zero — no opinion — when there is no live match, or when the bits coded
    /// so far have already contradicted it.
    #[inline]
    fn match_prediction(&self, c0: u32) -> i32 {
        if self.match_byte < 0 {
            return 0;
        }
        let mp = self.match_byte as u32;
        let bits_seen = 31 - c0.leading_zeros(); // 0..7
        // The bits already coded must be a prefix of the predicted byte.
        let coded = c0 - (1 << bits_seen);
        let expect = mp >> (8 - bits_seen);
        if coded != expect {
            return 0; // match contradicted within this byte
        }
        let predicted_bit = (mp >> (7 - bits_seen)) & 1;
        let conf = (400 + (self.match_len.min(28) as i32) * 58).min(2000);
        if predicted_bit == 1 {
            conf
        } else {
            -conf
        }
    }

    #[inline]
    pub fn update(&mut self, bit: i32) {
        if self.fast_mode {
            match self.fast_state {
                1 => {
                    // Adapt the match-confidence entry that was used.
                    let target = (bit << 16) as i32;
                    let v = self.fast_p[self.fast_idx] as i32;
                    self.fast_p[self.fast_idx] = (v + ((target - v) >> 5)) as u16;
                }
                _ => {
                    self.t0[self.idx0].update(bit);
                    self.t1[self.idx1].update(bit);
                }
            }
            self.c0 = (self.c0 << 1) | (bit as u32);
            return;
        }
        self.t0[self.idx0].update(bit);
        self.t1[self.idx1].update(bit);

        // Bit-history models: adapt the state-map entry that was used, then
        // advance the node's state by the observed bit.
        let sidx = (self.nib_path - 1) as usize;
        let smp = self.sm_plane();
        for k in 0..self.nbh {
            let s = self.bh_state[k];
            self.bh_sm[k][smp + s as usize].update(bit);
            self.bh[k].t[self.bh_off[k] + sidx] = state_next(s, bit);
        }
        self.nib_path = (self.nib_path << 1) | (bit as u32);

        // First-layer weights: gradient step on coding error for all views.
        let err = ((bit << 12) - self.pr) * self.mix_lr;
        let avx2 = self.use_avx2;
        Self::train(&mut self.wa, self.ctx_a, &self.tx, err, avx2);
        if !self.turbo {
            Self::train(&mut self.wb, self.ctx_b, &self.tx, err, avx2);
            Self::train(&mut self.wc, self.ctx_c, &self.tx, err, avx2);
            Self::train(&mut self.we, self.ctx_e, &self.tx, err, avx2);
            Self::train(&mut self.wg, self.ctx_g, &self.tx, err, avx2);
        }
        Self::train(&mut self.wd, self.ctx_d, &self.tx, err, avx2);

        // Second-layer weights: same error, over the first-layer outputs.
        // The product is widened to i64: |mi| can reach 2^15 and |err| 2^14.3,
        // so the i32 form could overflow and corrupt the combiner.
        let base = self.ctx_f * NMIX;
        for i in 0..NMIX {
            let g = (((self.mi[i] as i64) * (err as i64) + 0x8000) >> 16) as i32;
            let nw = self.wf[base + i] + g;
            self.wf[base + i] = nw.clamp(-(1 << 20), 1 << 20);
        }

        self.apm0.update(bit);
        self.apm2.update(bit);
        if !self.turbo {
            self.apm1.update(bit);
            self.apm3.update(bit);
        }

        self.c0 = (self.c0 << 1) | (bit as u32);

        // The high nibble just completed: the low nibble's bucket addresses
        // are now known, so start their cache lines moving immediately. The
        // find() pass only runs in the next predict() call, after this bit's
        // APM updates — that slack hides most of the memory latency.
        if self.c0 >> 4 == 1 {
            let hs = self.nib1_hashes(self.c0 & 15);
            for k in 0..self.nbh {
                self.bh[k].prefetch(hs[k]);
            }
            self.pending_hs = hs;
        }
    }

    #[inline]
    fn train(w: &mut [i32], ctx: usize, tx: &[i32; NINP], err: i32, use_avx2: bool) {
        let base = ctx * NINP;
        let row = &mut w[base..base + NINP];
        #[cfg(target_arch = "x86_64")]
        if use_avx2 {
            // SAFETY: only taken when AVX2 was detected at runtime.
            unsafe { train_avx2(row, tx, err) };
            return;
        }
        #[cfg(not(target_arch = "x86_64"))]
        let _ = use_avx2;
        train_scalar(row, tx, err);
    }

    /// Commit a finished byte: update history, the match model, and rebuild
    /// per-byte context hashes.
    #[inline]
    pub fn next_byte(&mut self, byte: u8) {
        // --- match model bookkeeping ------------------------------------
        // Did the active match correctly predict this byte?
        if self.match_byte == byte as i32 && self.match_len > 0 {
            self.match_len += 1;
            self.match_ptr += 1;
        } else {
            self.match_len = 0;
        }

        self.buf.push(byte);
        let pos = self.buf.len();

        // Refresh / seed a match from the suffix hashes. On a fresh seed we
        // *verify* the candidate by extending backward, both to reject hash
        // collisions and to recover the true match length — long verified
        // matches let the mixer predict the continuation near-certainly, which
        // is what captures long-range / cross-copy redundancy in big archives.
        // The 8-byte hash is tried before the 4-byte hash: a longer anchor is
        // both less likely to collide and likelier to continue correctly.
        let mut cand_long = MATCH_EMPTY;
        if pos >= MATCH_MIN_LONG {
            let h = (self.suffix_hash_n(MATCH_MIN_LONG) & self.match_mask) as usize;
            cand_long = self.match_table_long[h];
            self.match_table_long[h] = pos as u32;
        }
        if pos >= MATCH_MIN {
            let h = (self.suffix_hash_n(MATCH_MIN) & self.match_mask) as usize;
            let cand_short = self.match_table[h];
            self.match_table[h] = pos as u32;
            if self.match_len == 0 {
                // Verify both candidates and keep the *longer* match: a
                // deeper anchor predicts the continuation with more
                // confidence, and the two hashes often disagree.
                let mut best_ptr = 0usize;
                let mut best_len = 0usize;
                for cand in [cand_long, cand_short] {
                    if cand != MATCH_EMPTY && (cand as usize) < pos && cand as usize != best_ptr {
                        let c = cand as usize;
                        let max = c.min(pos);
                        let mut l = 0usize;
                        while l < max && self.buf[c - 1 - l] == self.buf[pos - 1 - l] && l < 0xffff
                        {
                            l += 1;
                        }
                        if l > best_len {
                            best_ptr = c;
                            best_len = l;
                        }
                    }
                }
                if best_len >= MATCH_MIN {
                    self.match_ptr = best_ptr;
                    self.match_len = best_len as u32;
                }
            }
        }
        self.match_byte = if self.match_len > 0 && self.match_ptr < self.buf.len() {
            self.buf[self.match_ptr] as i32
        } else {
            -1
        };

        // Two-speed switch for the upcoming byte: deep inside a verified
        // match, code it on the fast path. Both sides compute this from the
        // same decoded history, so the choice never needs to be signalled.
        self.fast_mode = self.match_len >= self.fast_len && self.match_byte >= 0;

        // --- indirect bookkeeping -----------------------------------------
        // Record the byte that just followed the previous order-2/order-3/
        // order-4 contexts (slots were resolved when those contexts were
        // current).
        self.ind2[self.ind2_idx] = byte;
        self.ind3[self.ind3_idx] = byte;
        self.ind4[self.ind4_idx] = byte;
        self.indt[self.indt_idx] = byte;

        // --- markup state -------------------------------------------------
        self.update_markup(byte);

        // --- context history --------------------------------------------
        self.hist.copy_within(0..15, 1);
        self.hist[0] = byte;
        self.c0 = 1;

        if byte.is_ascii_alphabetic() {
            let lower = byte | 0x20;
            self.word_hash = self
                .word_hash
                .wrapping_add(lower as u32 + 1)
                .wrapping_mul(PR1);
        } else {
            if self.word_hash != 0 {
                self.prev_word = self.last_word;
                self.last_word = self.word_hash;
            }
            self.word_hash = 0;
        }

        // The first NBH_TURBO models are the turbo profile's entire roster,
        // so the reduced profile is simply a prefix of the full one.
        self.bh_base[0] = hash_ctx(&self.hist[0..2], 2); // order-2
        self.bh_base[1] = hash_ctx(&self.hist[0..3], 3); // order-3
        self.bh_base[2] = hash_ctx(&self.hist[0..4], 4); // order-4
        self.bh_base[3] = hash_ctx(&self.hist[0..5], 5); // order-5
        self.bh_base[4] = self.word_hash.wrapping_mul(PR1) ^ 0xABCD_1234; // word
        if self.nbh > NBH_TURBO {
            self.bh_base[5] = hash_ctx(&self.hist[0..6], 6); // order-6
            self.bh_base[6] = hash_ctx(&self.hist[0..7], 7); // order-7
            // Word-pair: previous finished word + current word prefix. Models
            // bigram structure in natural-language text ("of the", "in a").
            self.bh_base[7] = self
                .last_word
                .wrapping_mul(PR2)
                .wrapping_add(self.word_hash)
                .wrapping_mul(PR1)
                ^ 0x5A5A_C3C3;
            // Sparse contexts: skip-grams and high-nibble views — useful on
            // structured binary where the low bits are noise.
            self.bh_base[8] = hash_ctx(&self.hist[1..2], 23);
            self.bh_base[9] = hash_ctx(&[self.hist[0] & 0xF0, self.hist[1] & 0xF0], 29);
            self.bh_base[10] = hash_ctx(&[self.hist[0], self.hist[2]], 31);
            self.bh_base[11] = hash_ctx(&[self.hist[1], self.hist[2]], 37);
            self.bh_base[12] = hash_ctx(&[self.hist[0], self.hist[3]], 41);
            self.bh_base[13] = hash_ctx(&self.hist[2..4], 43);

            // Stride bases: predict the upcoming byte (at index buf.len())
            // from the same lane of previous samples `stride` bytes back.
            let n = self.buf.len();
            for (k, &s) in STRIDES.iter().enumerate() {
                let b1 = if n >= s { self.buf[n - s] as u32 } else { 0 };
                let b2 = if n >= 2 * s { self.buf[n - 2 * s] as u32 } else { 0 };
                let mut h = (s as u32).wrapping_mul(PR1).wrapping_add(0x55AA_33CC);
                h = (h ^ (b1 + 1)).wrapping_mul(PR1);
                h = (h ^ (b2 + 1)).wrapping_mul(PR1);
                self.bh_base[NHASH + NSPARSE + k] = h ^ (h >> 15);
            }

            // Indirect contexts: resolve the slot for the *new* order-2 /
            // order-3 context, read the byte that followed it last time, and
            // fold that byte into the model context. The committed byte will
            // be written back into these same slots on the next call.
            self.ind2_idx = ((self.hist[1] as usize) << 8) | self.hist[0] as usize;
            self.ind3_idx = (hash_ctx(&self.hist[0..3], 53) & self.ind3_mask) as usize;
            self.ind4_idx = (hash_ctx(&self.hist[0..4], 83) & self.ind4_mask) as usize;
            let b2 = self.ind2[self.ind2_idx];
            let b3 = self.ind3[self.ind3_idx];
            let b4 = self.ind4[self.ind4_idx];
            let ind_base = NHASH + NSPARSE + NSTRIDE;
            self.bh_base[ind_base] = hash_ctx(&[b2, self.hist[0]], 47);
            self.bh_base[ind_base + 1] = hash_ctx(&[b3, self.hist[0], self.hist[1]], 59);
            self.bh_base[ind_base + 2] = hash_ctx(&[b4, self.hist[0], self.hist[1]], 73);

            // Text contexts: high orders bridge the gap between order-7 and
            // the match model (Wikipedia markup repeats at 8-12 byte scale),
            // and a case-folded order-3 merges "The"/"the" statistics.
            let text_base = ind_base + NIND;
            self.bh_base[text_base] = hash_ctx(&self.hist[0..8], 8);
            self.bh_base[text_base + 1] = hash_ctx(&self.hist[0..10], 10);
            // Very high orders bridge medium-range statistical structure
            // (repeated phrases, markup runs) that lies between order-10 and
            // the exact-match model — the mixer trusts them only where they
            // have evidence, so an empty high-order context costs nothing.
            self.bh_base[text_base + 2] = hash_ctx(&self.hist[0..12], 12);
            self.bh_base[text_base + 3] = hash_ctx(&self.hist[0..16], 16);
            let folded = [
                self.hist[0] | 0x20,
                self.hist[1] | 0x20,
                self.hist[2] | 0x20,
            ];
            self.bh_base[text_base + 4] = hash_ctx(&folded, 67);

            // --- markup contexts -----------------------------------------
            let mk = text_base + NTEXT;
            // Enclosing element + the byte that last followed this element
            // context, so `<title>` and `<timestamp>` content are modelled
            // apart and each carries its own "what usually comes next" cue.
            self.indt_idx = (self
                .tag_hash
                .wrapping_mul(PR2)
                .wrapping_add(self.hist[0] as u32)
                & self.indt_mask) as usize;
            let bt = self.indt[self.indt_idx];
            self.bh_base[mk] = hash_ctx(&[bt, self.hist[0], self.tag_hash as u8], 89)
                ^ self.tag_hash.wrapping_mul(PR1);
            // Innermost unclosed delimiter + depth + the previous byte.
            let d = self.nest_depth.min(7) as u8;
            let top = if self.nest_depth > 0 {
                self.nest[self.nest_depth - 1]
            } else {
                0
            };
            self.bh_base[mk + 1] = hash_ctx(&[top, d, self.hist[0]], 97);
            // Column and how the line started: indentation and list markers.
            self.bh_base[mk + 2] =
                hash_ctx(&[self.col.min(63) as u8, self.hist[0]], 101) ^ self.line_head;
            // Word trigram: the two previous whole words plus the prefix of
            // the word being written.
            self.bh_base[mk + 3] = self
                .prev_word
                .wrapping_mul(PR2)
                .wrapping_add(self.last_word)
                .wrapping_mul(PR1)
                .wrapping_add(self.word_hash)
                .wrapping_mul(PR2)
                ^ 0x3C3C_A5A5;
        }

        // First-layer `wc` / `we` selections: hashed order-3 and order-6,
        // fixed for the byte. The order-6 view lets the combiner trust a
        // different blend of models where a long local context is decisive
        // (repeated markup, boilerplate) than in generic order-3 contexts.
        self.ctx_c = (hash_ctx(&self.hist[0..3], 61) as usize) & self.mix_mask;
        self.ctx_e = (hash_ctx(&self.hist[0..6], 71) as usize) & self.mix_mask;
        // Word-level view: which word (if any) we are currently inside. Zero
        // between words, so all non-word positions share one weight vector.
        self.ctx_g = (self.word_hash.wrapping_mul(PR2) as usize) & self.mix_mask;

        // High-nibble bucket addresses are now known; start their lines early
        // (pointless when the next byte takes the fast path).
        if !self.fast_mode {
            for k in 0..self.nbh {
                self.bh[k].prefetch(self.bh_base[k]);
            }
        }
    }

    /// Advance the markup trackers by one committed byte: which element
    /// encloses us, how deep the bracket nesting is, and where we are in the
    /// line. Purely a function of the decoded prefix, so the decoder derives
    /// the identical state without any of it being transmitted.
    ///
    /// The scanners are deliberately forgiving — malformed or non-markup input
    /// just produces a stable, meaningless context that the mixer learns to
    /// ignore, rather than a parse error.
    #[inline]
    fn update_markup(&mut self, byte: u8) {
        // --- element scanner ---------------------------------------------
        match self.tag_scan {
            0 => {
                if byte == b'<' {
                    self.tag_scan = 1;
                    self.tag_name = 0;
                    self.tag_close = false;
                }
            }
            1 => {
                if byte == b'/' && self.tag_name == 0 {
                    self.tag_close = true;
                } else if byte.is_ascii_alphanumeric() || byte == b'_' || byte == b':' {
                    self.tag_name = self
                        .tag_name
                        .wrapping_add(byte as u32 + 1)
                        .wrapping_mul(PR1);
                } else {
                    // Name complete. A closing tag pops the stack; an opening
                    // tag pushes, unless it is self-closing (handled on '>').
                    if self.tag_close {
                        if self.tag_depth > 0 {
                            self.tag_depth -= 1;
                        }
                    } else if self.tag_depth < self.tag_stack.len() {
                        self.tag_stack[self.tag_depth] = self.tag_name;
                        self.tag_depth += 1;
                    }
                    self.tag_hash = if self.tag_depth > 0 {
                        self.tag_stack[self.tag_depth - 1]
                    } else {
                        0
                    };
                    self.tag_scan = if byte == b'>' { 0 } else { 2 };
                }
            }
            _ => {
                if byte == b'>' {
                    // `<br />`-style self-closing tag: undo the push.
                    if self.hist[0] == b'/' && self.tag_depth > 0 {
                        self.tag_depth -= 1;
                        self.tag_hash = if self.tag_depth > 0 {
                            self.tag_stack[self.tag_depth - 1]
                        } else {
                            0
                        };
                    }
                    self.tag_scan = 0;
                }
            }
        }

        // --- bracket / quote nesting --------------------------------------
        // Wiki markup nests `[[...]]`, `{{...}}` and `''...''` far more often
        // than it nests parentheses, and all of them change what follows.
        let open = matches!(byte, b'[' | b'{' | b'(');
        let close = matches!(byte, b']' | b'}' | b')');
        if open {
            if self.nest_depth < self.nest.len() {
                self.nest[self.nest_depth] = byte;
                self.nest_depth += 1;
            }
        } else if close && self.nest_depth > 0 {
            self.nest_depth -= 1;
        }

        // --- line shape ----------------------------------------------------
        if byte == b'\n' {
            self.col = 0;
            self.line_head = 0;
        } else {
            self.col += 1;
            // Only the first four bytes of a line define its shape; after that
            // the hash freezes so every position in the line shares one key.
            if self.col <= 4 {
                self.line_head = self
                    .line_head
                    .wrapping_add(byte as u32 + 1)
                    .wrapping_mul(PR2);
            }
        }
    }

    /// Hash of the last `len` committed bytes (salted by `len` so the short
    /// and long match tables never see compatible keys).
    #[inline]
    fn suffix_hash_n(&self, len: usize) -> u32 {
        let n = self.buf.len();
        let mut h: u32 = 0x811C_9DC5 ^ (len as u32).wrapping_mul(PR2);
        for &b in &self.buf[n - len..n] {
            h = (h ^ (b as u32)).wrapping_mul(PR1);
        }
        h ^= h >> 15;
        h
    }
}

// ---------------------------------------------------------------------------
// Mixer kernels. The AVX2 versions are bit-identical to the scalar ones:
// every weight-input product fits i32 exactly (|w| <= W_CLAMP < 2^19,
// |tx| <= 2047), the dot sums are widened to i64 (associative, no overflow),
// and srai/min/max match Rust's >> and clamp. An archive encoded on an AVX2
// machine therefore decodes identically on a non-AVX2 one and vice versa.
// ---------------------------------------------------------------------------

#[inline]
fn dot_scalar(row: &[i32], tx: &[i32; NINP]) -> i32 {
    let mut acc = 0i64;
    for i in 0..NINP {
        acc += (row[i] as i64) * (tx[i] as i64);
    }
    (acc >> 16) as i32
}

#[inline]
fn train_scalar(row: &mut [i32], tx: &[i32; NINP], err: i32) {
    for i in 0..NINP {
        let nw = row[i] + (((tx[i] * err) + 0x8000) >> 16);
        row[i] = nw.clamp(-W_CLAMP, W_CLAMP);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_avx2(row: &[i32], tx: &[i32; NINP]) -> i32 {
    use std::arch::x86_64::*;
    debug_assert!(row.len() >= NINP);
    let mut acc = _mm256_setzero_si256(); // 4 x i64 partial sums
    let mut i = 0;
    while i < NINP {
        let w = _mm256_loadu_si256(row.as_ptr().add(i) as *const __m256i);
        let x = _mm256_loadu_si256(tx.as_ptr().add(i) as *const __m256i);
        let p = _mm256_mullo_epi32(w, x); // exact: |w * x| < 2^30
        let lo = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(p));
        let hi = _mm256_cvtepi32_epi64(_mm256_extracti128_si256::<1>(p));
        acc = _mm256_add_epi64(acc, _mm256_add_epi64(lo, hi));
        i += 8;
    }
    let mut lanes = [0i64; 4];
    _mm256_storeu_si256(lanes.as_mut_ptr() as *mut __m256i, acc);
    ((lanes[0] + lanes[1] + lanes[2] + lanes[3]) >> 16) as i32
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn train_avx2(row: &mut [i32], tx: &[i32; NINP], err: i32) {
    use std::arch::x86_64::*;
    debug_assert!(row.len() >= NINP);
    let verr = _mm256_set1_epi32(err);
    let vround = _mm256_set1_epi32(0x8000);
    let vmax = _mm256_set1_epi32(W_CLAMP);
    let vmin = _mm256_set1_epi32(-W_CLAMP);
    let mut i = 0;
    while i < NINP {
        let x = _mm256_loadu_si256(tx.as_ptr().add(i) as *const __m256i);
        let p = _mm256_mullo_epi32(x, verr); // exact: |tx * err| < 2^26
        let d = _mm256_srai_epi32::<16>(_mm256_add_epi32(p, vround));
        let w = _mm256_loadu_si256(row.as_ptr().add(i) as *const __m256i);
        let nw = _mm256_add_epi32(w, d);
        let cl = _mm256_max_epi32(vmin, _mm256_min_epi32(vmax, nw));
        _mm256_storeu_si256(row.as_mut_ptr().add(i) as *mut __m256i, cl);
        i += 8;
    }
}

/// Hash a context byte slice with an order-specific salt.
#[inline]
fn hash_ctx(bytes: &[u8], salt: u32) -> u32 {
    let mut h = salt.wrapping_mul(PR1).wrapping_add(0x1234_5678);
    for &b in bytes {
        h = (h ^ (b as u32 + 1)).wrapping_mul(PR1);
        h ^= h >> 15;
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The AVX2 mixer kernels must match the scalar ones bit-for-bit, or
    /// archives would not decode across machines with different CPUs.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn simd_kernels_match_scalar() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return; // nothing to compare on this machine
        }
        let mut x: u64 = 0x1234_5678_9abc_def0;
        let mut rnd = move |m: i32| {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((x >> 33) as i32 % (2 * m + 1)) - m
        };
        for _ in 0..200 {
            let mut tx = [0i32; NINP];
            let mut row_a = vec![0i32; NINP];
            for i in 0..NINP {
                tx[i] = rnd(2047);
                row_a[i] = rnd(W_CLAMP);
            }
            let mut row_b = row_a.clone();
            let err = rnd(4095 * MIX_LR);

            let d_scalar = dot_scalar(&row_a, &tx);
            let d_avx2 = unsafe { dot_avx2(&row_a, &tx) };
            assert_eq!(d_scalar, d_avx2, "dot kernels diverged");

            train_scalar(&mut row_a, &tx, err);
            unsafe { train_avx2(&mut row_b, &tx, err) };
            assert_eq!(row_a, row_b, "train kernels diverged");
        }
    }
}
