//! Bit-level context-mixing predictor for the CPGC-NX engine (v7).
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

/// Pick a power-of-two table exponent appropriate for `n` input bytes.
fn table_bits(n: usize) -> u32 {
    // Aim for a table a few times larger than the input, clamped to range.
    let target = (usize::BITS - n.max(1).leading_zeros()) + 2;
    target.clamp(HBITS_MIN, HBITS_MAX)
}

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
// high-nibble), then the four stride contexts. Sparse and stride contexts
// are low-cardinality, so their tables are capped smaller.
const NHASH: usize = 8;
const NSPARSE: usize = 6;
const NBH: usize = NHASH + NSPARSE + NSTRIDE; // 18 bit-history models

// First-layer mixer inputs:
//   order-0 + order-1 dual counters (2 each)
// + NBH bit-history models * 2 (state map + direct state estimate)
// + 1 match model
// + 1 bias
const NIN: usize = 4 + NBH * 2 + 2;
const BH_IN: usize = 4; // first bit-history input index
const MATCH_IN: usize = BH_IN + NBH * 2;
const BIAS_IN: usize = NIN - 1;

// Second-layer mixer: combines the four first-layer outputs plus a bias.
const NMIX: usize = 5;
// Selected by (min(match_len, 7), bit position): the combiner learns, e.g.,
// to trust the match view less on low bits and counter views more there.
const NMIX_CTX: usize = 64;

// Mixer learning rate (scales the coding-error gradient applied to weights).
const MIX_LR: i32 = 5;

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
    idx: usize,
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
        Self { t, idx: 0 }
    }

    #[inline]
    fn refine(&mut self, pr: i32, cxt: usize) -> i32 {
        let s = (stretch(pr) + 2048).clamp(0, 4095);
        let j = (s >> 7) as usize;
        let w = s & 127;
        let base = cxt * 33 + j;
        self.idx = base + (w >> 6) as usize;
        let lo = self.t[base] as i32;
        let hi = self.t[base + 1] as i32;
        let p16 = (lo * (128 - w) + hi * w) >> 7;
        (p16 >> 4).clamp(1, 4095)
    }

    #[inline]
    fn update(&mut self, bit: i32) {
        let target = bit << 16;
        let v = self.t[self.idx] as i32;
        self.t[self.idx] = (v + ((target - v) >> 7)) as u16;
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
    hist: [u8; 8],
    word_hash: u32,
    last_word: u32, // hash of the most recently *finished* word

    // Bit-history models: nibble-bucketed state tables + per-model state maps.
    bh: Vec<BhTable>,            // NBH tables
    bh_sm: Vec<[SmEntry; 256]>,  // NBH state maps
    bh_base: [u32; NBH],         // per-byte context hashes
    bh_off: [usize; NBH],        // resolved bucket slot-array offsets
    bh_state: [u8; NBH],         // states read by predict(), for update()
    nib_path: u32,               // nibble-local path register (1..15)
    pending_hs: [u32; NBH],      // low-nibble hashes, prefetched a bit early

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

    // First-layer mixer: four context-selected weight sets.
    wa: Vec<i32>, // [256][NIN] selected by previous byte
    wb: Vec<i32>, // [64][NIN]  selected by match-length bucket
    wc: Vec<i32>, // [256][NIN] selected by the byte before previous
    wd: Vec<i32>, // [256][NIN] selected by the partial byte (c0)
    tx: [i32; NIN],
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
    pub fn new(n: usize) -> Self {
        let _ = squash(0);
        let _ = stretch(2048);
        let _ = st_direct_tbl();

        // Bucket counts: states are 1 byte (16-byte buckets serve a whole
        // nibble), so the tables are far smaller than v6's 6-byte-slot tables
        // at equal context capacity. Stride contexts are low-cardinality, so
        // their tables are capped harder to stay cache-friendly.
        let bh_bits = (table_bits(n).saturating_sub(3)).clamp(11, 19);
        let stride_bits = bh_bits.min(16);

        // The match tables store one u32 per slot.
        let mbits = table_bits(n).min(HBITS_MAX);
        let msize = 1usize << mbits;

        // Initialise the second-layer weights so the mixer starts out close to
        // an average of its four inputs; gradient descent refines from there.
        let avg_w = ((1i64 << 16) / (NMIX as i64 - 1)) as i32;
        let mut wf = vec![0i32; NMIX_CTX * NMIX];
        for c in 0..NMIX_CTX {
            for i in 0..NMIX - 1 {
                wf[c * NMIX + i] = avg_w;
            }
            // the bias weight (last slot) stays 0
        }

        Self {
            t0: vec![DualCounter::INIT; 256],
            t1: vec![DualCounter::INIT; 1 << 16],
            idx0: 0,
            idx1: 0,
            hist: [0; 8],
            word_hash: 0,
            last_word: 0,
            bh: (0..NBH)
                .map(|k| BhTable::new(if k < NHASH { bh_bits } else { stride_bits }))
                .collect(),
            bh_sm: vec![sm_init(); NBH],
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
            wa: vec![0i32; 256 * NIN],
            wb: vec![0i32; 64 * NIN],
            wc: vec![0i32; 256 * NIN],
            wd: vec![0i32; 256 * NIN],
            tx: [0; NIN],
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
            apm2: Apm::new(1024),
            apm3: Apm::new(256),
            final_pr: 2048,
        }
    }

    /// Locate every model's bucket for the nibble that starts now. `nib0` is
    /// `None` for the high nibble, or the four already-coded high bits for the
    /// low nibble. All candidate cache lines are prefetched before the find()
    /// pass so the (usually missing) loads overlap.
    /// Hashes for the low nibble's buckets, given the four high bits.
    #[inline]
    fn nib1_hashes(&self, nib0: u32) -> [u32; NBH] {
        let salt = (nib0 + 17).wrapping_mul(PR2);
        let mut hs = [0u32; NBH];
        for k in 0..NBH {
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
        for k in 0..NBH {
            self.bh_off[k] = self.bh[k].find(hs[k]);
        }
        self.nib_path = 1;
    }

    #[inline]
    pub fn predict(&mut self) -> i32 {
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
        for k in 0..NBH {
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

        // --- first-layer mixing: four context-selected weight sets -------
        self.ctx_a = self.hist[0] as usize;
        self.ctx_b = (self.match_len.min(63)) as usize;
        self.ctx_c = self.hist[1] as usize;
        self.ctx_d = (c0 & 0xff) as usize;
        let sa = self.dot(&self.wa, self.ctx_a);
        let sb = sa;
        let sc = sa;
        let sd = sa;

        // --- second-layer mixing: a small learned combiner ---------------
        self.mi = [sa, sb, sc, sd, 256];
        let bits_seen = (31 - c0.leading_zeros()) as usize; // 0..7
        self.ctx_f = (self.match_len.min(7) as usize) << 3 | bits_seen;
        let mixed = self.dot2(self.ctx_f);
        self.pr = squash(mixed);

        // --- SSE refinement ---------------------------------------------
        let p0 = self.apm0.refine(self.pr, self.ctx_d);
        let mut p = (self.pr + p0 * 3) >> 2;
        let p1 = self.apm1.refine(p, self.ctx_a);
        p = (p + p1 * 3) >> 2;
        let p2 = self.apm2.refine(p, (self.bh_base[1] & 0x3ff) as usize);
        p = (p + p2 * 3) >> 2;
        let p3 = self.apm3.refine(p, self.ctx_c);
        p = (p + p3 * 3) >> 2;
        self.final_pr = p.clamp(1, 4095);
        self.final_pr
    }

    #[inline]
    fn dot(&self, w: &[i32], ctx: usize) -> i32 {
        let base = ctx * NIN;
        let mut acc = 0i64;
        for i in 0..NIN {
            acc += (w[base + i] as i64) * (self.tx[i] as i64);
        }
        (acc >> 16) as i32
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
        self.t0[self.idx0].update(bit);
        self.t1[self.idx1].update(bit);

        // Bit-history models: adapt the state-map entry that was used, then
        // advance the node's state by the observed bit.
        let sidx = (self.nib_path - 1) as usize;
        for k in 0..NBH {
            let s = self.bh_state[k];
            self.bh_sm[k][s as usize].update(bit);
            self.bh[k].t[self.bh_off[k] + sidx] = state_next(s, bit);
        }
        self.nib_path = (self.nib_path << 1) | (bit as u32);

        // First-layer weights: gradient step on coding error for all views.
        let err = ((bit << 12) - self.pr) * MIX_LR;
        Self::train(&mut self.wa, self.ctx_a, &self.tx, err);
        
        
        

        // Second-layer weights: same error, over the four first-layer outputs.
        let base = self.ctx_f * NMIX;
        for i in 0..NMIX {
            let nw = self.wf[base + i] + (((self.mi[i] * err) + 0x8000) >> 16);
            self.wf[base + i] = nw.clamp(-(1 << 20), 1 << 20);
        }

        self.apm0.update(bit);
        self.apm1.update(bit);
        self.apm2.update(bit);
        self.apm3.update(bit);

        self.c0 = (self.c0 << 1) | (bit as u32);

        // The high nibble just completed: the low nibble's bucket addresses
        // are now known, so start their cache lines moving immediately. The
        // find() pass only runs in the next predict() call, after this bit's
        // APM updates — that slack hides most of the memory latency.
        if self.c0 >> 4 == 1 {
            let hs = self.nib1_hashes(self.c0 & 15);
            for k in 0..NBH {
                self.bh[k].prefetch(hs[k]);
            }
            self.pending_hs = hs;
        }
    }

    #[inline]
    fn train(w: &mut [i32], ctx: usize, tx: &[i32; NIN], err: i32) {
        let base = ctx * NIN;
        for i in 0..NIN {
            let nw = w[base + i] + (((tx[i] * err) + 0x8000) >> 16);
            w[base + i] = nw.clamp(-(1 << 20), 1 << 20);
        }
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
                for cand in [cand_long, cand_short] {
                    if cand != MATCH_EMPTY && (cand as usize) < pos {
                        let c = cand as usize;
                        let max = c.min(pos);
                        let mut l = 0usize;
                        while l < max && self.buf[c - 1 - l] == self.buf[pos - 1 - l] && l < 0xffff
                        {
                            l += 1;
                        }
                        if l >= MATCH_MIN {
                            self.match_ptr = c;
                            self.match_len = l as u32;
                            break;
                        }
                    }
                }
            }
        }
        self.match_byte = if self.match_len > 0 && self.match_ptr < self.buf.len() {
            self.buf[self.match_ptr] as i32
        } else {
            -1
        };

        // --- context history --------------------------------------------
        self.hist.copy_within(0..7, 1);
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

        self.bh_base[0] = hash_ctx(&self.hist[0..2], 2); // order-2
        self.bh_base[1] = hash_ctx(&self.hist[0..3], 3); // order-3
        self.bh_base[2] = hash_ctx(&self.hist[0..4], 4); // order-4
        self.bh_base[3] = hash_ctx(&self.hist[0..5], 5); // order-5
        self.bh_base[4] = hash_ctx(&self.hist[0..6], 6); // order-6
        self.bh_base[5] = hash_ctx(&self.hist[0..7], 7); // order-7
        self.bh_base[6] = self.word_hash.wrapping_mul(PR1) ^ 0xABCD_1234; // word
        // Word-pair: previous finished word + current word prefix. Models
        // bigram structure in natural-language text ("of the", "in a", ...).
        self.bh_base[7] = self
            .last_word
            .wrapping_mul(PR2)
            .wrapping_add(self.word_hash)
            .wrapping_mul(PR1)
            ^ 0x5A5A_C3C3;
        // Sparse contexts: a skip-gram that ignores the immediately previous
        // byte, and the high nibbles of the last two bytes — both useful on
        // structured binary where the low bits are noise.
        self.bh_base[8] = hash_ctx(&self.hist[1..2], 23);
        self.bh_base[9] = hash_ctx(&[self.hist[0] & 0xF0, self.hist[1] & 0xF0], 29);
        self.bh_base[10] = hash_ctx(&[self.hist[0], self.hist[2]], 31);
        self.bh_base[11] = hash_ctx(&[self.hist[1], self.hist[2]], 37);
        self.bh_base[12] = hash_ctx(&[self.hist[0], self.hist[3]], 41);
        self.bh_base[13] = hash_ctx(&self.hist[2..4], 43);

        // Stride bases: predict the upcoming byte (at index buf.len()) from the
        // same lane of the previous one/two samples `stride` bytes back.
        let n = self.buf.len();
        for (k, &s) in STRIDES.iter().enumerate() {
            let b1 = if n >= s { self.buf[n - s] as u32 } else { 0 };
            let b2 = if n >= 2 * s { self.buf[n - 2 * s] as u32 } else { 0 };
            let mut h = (s as u32).wrapping_mul(PR1).wrapping_add(0x55AA_33CC);
            h = (h ^ (b1 + 1)).wrapping_mul(PR1);
            h = (h ^ (b2 + 1)).wrapping_mul(PR1);
            self.bh_base[NHASH + NSPARSE + k] = h ^ (h >> 15);
        }

        // High-nibble bucket addresses are now known; start their lines early.
        for k in 0..NBH {
            self.bh[k].prefetch(self.bh_base[k]);
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
