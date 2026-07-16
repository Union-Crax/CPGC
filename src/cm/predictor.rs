//! Bit-level context-mixing predictor for the CPGC-NX engine (v8).
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
//!   a two-candidate replacement policy evicts the less-established bucket,
//!   so colliding contexts no longer silently corrupt each other. All
//!   candidate lines are prefetched at the nibble boundary so the misses
//!   overlap instead of serialising.
//! * **Dual long-match model.** Two rolling hashes (8-byte and 4-byte suffix)
//!   point at the most recent place the current suffix occurred — the longer
//!   hash is preferred so seeds start from more reliable anchors — and the
//!   predictor forecasts the *bit* of the historical continuation with
//!   confidence that grows with verified match length.
//! * **Two-layer logistic mixer.** A first layer holds four independently
//!   context-selected weight vectors (by previous byte, by the byte before it,
//!   by match-length bucket, and by the partial byte being decoded); a small
//!   learned second layer selected by (match length, bit position) combines
//!   their stretched outputs, trained online by gradient descent.
//! * **Chained SSE.** Four adaptive probability maps (keyed by partial byte,
//!   previous bytes, and an order-3 hash) refine the result before the binary
//!   arithmetic coder.
//! * **Two-speed coding.** Bytes deep inside a verified match (>= FAST_LEN)
//!   are coded by a tiny match-confidence SSE instead of the full model —
//!   deterministically, since both sides track the match length — making
//!   redundant regions nearly free in time as well as bits.
//! * **Runtime-SIMD mixer.** The first-layer dot products and weight updates
//!   run on AVX2 when available, with a bit-identical scalar fallback, so
//!   the bitstream never depends on the CPU.
//! * **Two profiles.** Turbo (levels 1-3) runs a 5-model prefix of the
//!   roster with two mixer views and two APMs; full runs everything. The
//!   profile is recorded in the payload header.

use std::sync::OnceLock;

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
// 18-model mixer — the byte is almost certainly the match continuation, so
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
    let plus = (mem >= MEM_PLUS) as u32;
    let hash_bits = if mem >= MEM_BIG {
        raw_bits(n).clamp(11, 23 + plus)
    } else {
        raw_bits(n).clamp(HBITS_MIN, HBITS_MAX).saturating_sub(3).clamp(11, 19)
    };
    match MODEL_KIND[k] {
        Kind::Hash => hash_bits,
        Kind::Sparse => hash_bits.min(if mem >= MEM_BIG { 21 + plus } else { 16 }),
        Kind::Stride => hash_bits.min(16),
        Kind::Ind => hash_bits.min(if mem >= MEM_BIG { 22 + plus } else { 18 }),
    }
}

// Memory profiles (recorded in the payload so decode always agrees).
pub const MEM_STD: u8 = 0;
pub const MEM_BIG: u8 = 1; // levels 7+: up to 2^23-bucket hash tables
pub const MEM_PLUS: u8 = 2; // levels 8-9: up to 2^24 buckets, 2^25 match slots

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

/// A fresh state map, with every entry seeded from its state's closed-form
/// estimate rather than 0.5 — a brand-new context predicts sensibly from its
/// very first visit, and the count-adaptive rate then refines from there.
fn sm_init() -> [SmEntry; 256] {
    let mut t = [SmEntry { p: 32768, cnt: 0 }; 256];
    for (s, e) in t.iter_mut().enumerate() {
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
}

impl BhTable {
    fn new(bucket_bits: u32) -> Self {
        Self {
            t: vec![0u8; BUCKET << bucket_bits],
            mask: (1u32 << bucket_bits) - 1,
        }
    }

    /// Hint both candidate buckets for `h` into cache (they are usually the
    /// same 64-byte line). Called for every model *before* the find() pass so
    /// the memory latencies overlap.
    #[inline]
    fn prefetch(&self, h: u32) {
        let i0 = ((h & self.mask) as usize) * BUCKET;
        prefetch_ptr(unsafe { self.t.as_ptr().add(i0) });
        prefetch_ptr(unsafe { self.t.as_ptr().add(i0 ^ BUCKET) });
    }

    /// Find (or allocate) the bucket for hash `h`; returns the byte offset of
    /// its 15-state slot array. Two candidate buckets are probed; on a double
    /// miss the bucket whose root state carries less evidence is recycled.
    #[inline]
    fn find(&mut self, h: u32) -> usize {
        let cs = ((h >> 24) as u8) | 1; // 0 marks "never used"
        let i0 = ((h & self.mask) as usize) * BUCKET;
        let i1 = i0 ^ BUCKET;
        if self.t[i0] == cs {
            return i0 + 1;
        }
        if self.t[i1] == cs {
            return i1 + 1;
        }
        let k = if self.t[i0] == 0 {
            i0
        } else if self.t[i1] == 0 {
            i1
        } else {
            let e0 = self.t[i0 + 1];
            let e1 = self.t[i1 + 1];
            // total observations at the root slot = how established the bucket is
            if (e0 >> 4) + (e0 & 15) <= (e1 >> 4) + (e1 & 15) {
                i0
            } else {
                i1
            }
        };
        self.t[k] = cs;
        self.t[k + 1..k + BUCKET].fill(0);
        k + 1
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
const NIND: usize = 2;
const NTEXT: usize = 3; // order-8, order-10, case-folded order-3
const NBH: usize = NHASH + NSPARSE + NSTRIDE + NIND + NTEXT; // 23 bit-history models

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
    Kind::Ind, Kind::Ind,
    Kind::Hash, Kind::Hash,                         // orders 8, 10
    Kind::Hash,                                     // case-folded order-3
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
// order-2 context (2048 rows) rather than the single previous-previous byte:
// on text, which pair of bytes precedes the position is a far sharper cue
// for which models to trust than either byte alone.
const WC_ROWS: usize = 2048;

// Second-layer mixer: combines the four first-layer outputs plus a bias.
const NMIX: usize = 5;
// Selected by (is-text flag, min(match_len, 7), bit position): the combiner
// learns, e.g., to trust the match view less on low bits and counter views
// more there — with separate weights inside words vs elsewhere.
const NMIX_CTX: usize = 128;

// Mixer learning rate (scales the coding-error gradient applied to weights).
const MIX_LR: i32 = 5;
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
        self.t[self.base] = (lo + (((target - lo) * (128 - self.w)) >> 14)) as u16;
        self.t[self.base + 1] = (hi + (((target - hi) * self.w) >> 14)) as u16;
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
    last_word: u32, // hash of the most recently *finished* word

    // Indirect context state: per-context "byte that followed last time".
    // ind2 is direct-indexed by the order-2 context (collision-free); ind3
    // is indexed by a hashed order-3 context. A collision only yields a
    // noisy context input — never an incorrect decode.
    ind2: Vec<u8>,
    ind3: Vec<u8>,
    ind3_mask: u32,
    ind2_idx: usize, // slot to write the *next* committed byte into
    ind3_idx: usize,

    // Bit-history models: nibble-bucketed state tables + per-model state maps.
    bh: Vec<BhTable>,            // NBH tables
    bh_sm: Vec<[SmEntry; 256]>,  // NBH state maps
    bh_base: [u32; NBH],         // per-byte context hashes
    bh_off: [usize; NBH],        // resolved bucket slot-array offsets
    bh_state: [u8; NBH],         // states read by predict(), for update()
    nib_path: u32,               // nibble-local path register (1..15)
    pending_hs: [u32; NBH],      // low-nibble hashes, prefetched a bit early
    nbh: usize,                  // active model count (NBH_TURBO or NBH)
    turbo: bool,                 // reduced mixer/SSE roster for low levels

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
    fast_mode: bool,
    fast_state: u8,
    fast_idx: usize,
    fast_p: [u16; 256],

    // First-layer mixer: four context-selected weight sets.
    wa: Vec<i32>, // [256][NINP] selected by previous byte
    wb: Vec<i32>, // [64][NINP]  selected by match-length bucket
    wc: Vec<i32>, // [256][NINP] selected by the byte before previous
    wd: Vec<i32>, // [256][NINP] selected by the partial byte (c0)
    tx: [i32; NINP],
    use_avx2: bool, // AVX2 detected at runtime (paths are bit-identical)
    ctx_a: usize,
    ctx_b: usize,
    ctx_c: usize,
    ctx_d: usize,

    // Second-layer mixer: combines the four first-layer outputs.
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

        let nbh = if turbo { NBH_TURBO } else { NBH };

        // The match tables store one u32 per slot; the big profiles grow
        // them so long-range matches on a 100 MB+ segment survive (raw_bits,
        // not table_bits: the standard clamp must not cap the big profiles).
        let mbits = if mem >= MEM_BIG {
            raw_bits(n).clamp(HBITS_MIN, 24 + (mem >= MEM_PLUS) as u32)
        } else {
            table_bits(n)
        };
        let msize = 1usize << mbits;

        let ind3_bits: u32 = if mem >= MEM_BIG { 22 } else { 20 };

        // Initialise the second-layer weights so the mixer starts out close to
        // an average of its active view inputs (two in turbo, four in full);
        // gradient descent refines from there.
        let views = if turbo { 2 } else { 4 };
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
            ind2: vec![0u8; 1 << 16],
            ind3: vec![0u8; 1 << ind3_bits],
            ind3_mask: (1u32 << ind3_bits) - 1,
            ind2_idx: 0,
            ind3_idx: 0,
            bh: (0..nbh)
                .map(|k| BhTable::new(model_bits(k, n, mem)))
                .collect(),
            bh_sm: vec![sm_init(); nbh],
            nbh,
            turbo,
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
            wc: vec![0i32; WC_ROWS * NINP],
            wd: vec![0i32; 256 * NINP],
            tx: [0; NINP],
            #[cfg(target_arch = "x86_64")]
            use_avx2: std::arch::is_x86_feature_detected!("avx2"),
            #[cfg(not(target_arch = "x86_64"))]
            use_avx2: false,
            ctx_a: 0,
            ctx_b: 0,
            ctx_c: 0,
            ctx_d: 0,
            wf,
            mi: [0; NMIX],
            ctx_f: 0,
            pr: 2048,
            apm0: Apm::new(256),
            apm1: Apm::new(256),
            apm2: Apm::new(16384),
            apm3: Apm::new(WC_ROWS),
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
        let st_direct = st_direct_tbl();
        for k in 0..self.nbh {
            let s = self.bh[k].t[self.bh_off[k] + sidx];
            self.bh_state[k] = s;
            let e = self.bh_sm[k][s as usize];
            self.tx[BH_IN + k * 2] = stretch((e.p >> 4) as i32);
            self.tx[BH_IN + k * 2 + 1] = st_direct[s as usize] as i32;
        }

        // --- match model -------------------------------------------------
        self.tx[MATCH_IN] = self.match_prediction(c0);

        // --- bias --------------------------------------------------------
        self.tx[BIAS_IN] = 256;

        // --- first-layer mixing: context-selected weight sets -------------
        // (turbo runs only the prev-byte and partial-byte views; ctx_c is a
        // hashed order-2 selection, computed once per byte in next_byte)
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
        let (sb, sc) = if self.turbo {
            (0, 0)
        } else {
            (
                self.dot(&self.wb, self.ctx_b).clamp(-S_CLAMP, S_CLAMP),
                self.dot(&self.wc, self.ctx_c).clamp(-S_CLAMP, S_CLAMP),
            )
        };

        // --- second-layer mixing: a small learned combiner ---------------
        self.mi = [sa, sb, sc, sd, 256];
        let bits_seen = (31 - c0.leading_zeros()) as usize; // 0..7
        let in_word = self.hist[0].is_ascii_alphabetic() as usize;
        self.ctx_f = in_word << 6 | (self.match_len.min(7) as usize) << 3 | bits_seen;
        let mixed = self.dot2(self.ctx_f);
        self.pr = squash(mixed);

        // --- SSE refinement (turbo keeps apm0 + apm2 only) ----------------
        let p0 = self.apm0.refine(self.pr, self.ctx_d);
        let mut p = (self.pr + p0 * 3) >> 2;
        if !self.turbo {
            let p1 = self.apm1.refine(p, self.ctx_a);
            p = (p + p1 * 3) >> 2;
        }
        let p2 = self.apm2.refine(p, (self.bh_base[1] & 0x3fff) as usize);
        p = (p + p2 * 3) >> 2;
        if !self.turbo {
            let p3 = self.apm3.refine(p, self.ctx_c);
            p = (p + p3 * 3) >> 2;
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

    /// Stretched prediction from the match model for the current partial byte.
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
        for k in 0..self.nbh {
            let s = self.bh_state[k];
            self.bh_sm[k][s as usize].update(bit);
            self.bh[k].t[self.bh_off[k] + sidx] = state_next(s, bit);
        }
        self.nib_path = (self.nib_path << 1) | (bit as u32);

        // First-layer weights: gradient step on coding error for all views.
        let err = ((bit << 12) - self.pr) * MIX_LR;
        let avx2 = self.use_avx2;
        Self::train(&mut self.wa, self.ctx_a, &self.tx, err, avx2);
        if !self.turbo {
            Self::train(&mut self.wb, self.ctx_b, &self.tx, err, avx2);
            Self::train(&mut self.wc, self.ctx_c, &self.tx, err, avx2);
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
        self.fast_mode = self.match_len >= FAST_LEN && self.match_byte >= 0;

        // --- indirect bookkeeping -----------------------------------------
        // Record the byte that just followed the previous order-2/order-3
        // contexts (slots were resolved when those contexts were current).
        self.ind2[self.ind2_idx] = byte;
        self.ind3[self.ind3_idx] = byte;

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
            let b2 = self.ind2[self.ind2_idx];
            let b3 = self.ind3[self.ind3_idx];
            let ind_base = NHASH + NSPARSE + NSTRIDE;
            self.bh_base[ind_base] = hash_ctx(&[b2, self.hist[0]], 47);
            self.bh_base[ind_base + 1] = hash_ctx(&[b3, self.hist[0], self.hist[1]], 59);

            // Text contexts: high orders bridge the gap between order-7 and
            // the match model (Wikipedia markup repeats at 8-12 byte scale),
            // and a case-folded order-3 merges "The"/"the" statistics.
            let text_base = ind_base + NIND;
            self.bh_base[text_base] = hash_ctx(&self.hist[0..8], 8);
            self.bh_base[text_base + 1] = hash_ctx(&self.hist[0..10], 10);
            let folded = [
                self.hist[0] | 0x20,
                self.hist[1] | 0x20,
                self.hist[2] | 0x20,
            ];
            self.bh_base[text_base + 2] = hash_ctx(&folded, 67);
        }

        // First-layer `wc` selection: hashed order-2, fixed for the byte.
        self.ctx_c = (hash_ctx(&self.hist[0..2], 61) as usize) & (WC_ROWS - 1);

        // High-nibble bucket addresses are now known; start their lines early
        // (pointless when the next byte takes the fast path).
        if !self.fast_mode {
            for k in 0..self.nbh {
                self.bh[k].prefetch(self.bh_base[k]);
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
