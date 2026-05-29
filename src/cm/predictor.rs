//! Bit-level context-mixing predictor for the CPGC-NX engine.
//!
//! Produces `P(next bit == 1)` as a 12-bit probability. The architecture is a
//! new *combination* tuned for this codec rather than a port of any single
//! existing compressor:
//!
//! * **Dual learning-rate counters.** Every context slot stores *two* adaptive
//!   probabilities — one fast, one slow — and both are fed to the mixer. The
//!   fast counter reacts to local change; the slow one captures the stationary
//!   estimate. Letting the mixer weigh them per context recovers most of the
//!   benefit of an explicit run/confidence count at half the bookkeeping.
//! * **Long-match model.** A rolling hash points at the most recent place the
//!   current suffix occurred; the predictor then forecasts the *bit* of the
//!   historical continuation with confidence that grows with match length.
//!   This is what lets structured/repetitive data fall well below 1 bpb.
//! * **Two-context mixing layer.** Predictions are mixed by two weight sets
//!   selected by *different* contexts (previous byte, and match-length bucket)
//!   and averaged in the logistic domain — a cheap two-view mixer that beats a
//!   single weight selector.
//! * **Chained SSE.** Two adaptive probability maps refine the result.

use std::sync::OnceLock;

// Hashing multipliers (odd, good avalanche).
const PR1: u32 = 0x9E37_79B1;
const PR2: u32 = 0x85EB_CA77;

// Hashed-model tables are sized to the input (capped at 2^22 dual counters =
// 16 MiB each) so small inputs allocate little and large inputs get the full
// table. The size is derived deterministically from the byte count, which both
// encoder and decoder know, so the two sides always agree.
const HBITS_MAX: u32 = 22;
const HBITS_MIN: u32 = 14;

const MATCH_MIN: usize = 4; // suffix length that seeds a new match
const MATCH_EMPTY: u32 = u32::MAX;

/// Pick a power-of-two table exponent appropriate for `n` input bytes.
fn table_bits(n: usize) -> u32 {
    // Aim for a table a few times larger than the input, clamped to range.
    let target = (usize::BITS - n.max(1).leading_zeros()) + 2;
    target.clamp(HBITS_MIN, HBITS_MAX)
}

// Counter adaptation rates (fast / slow). Stride models use a middle rate.
const RATE_FAST: i32 = 3;
const RATE_SLOW: i32 = 6;
const RATE_STRIDE: i32 = 4;

/// Adapt a single 16-bit `P(bit==1)` counter toward the observed bit.
#[inline]
fn counter_update(slot: &mut u16, bit: i32) {
    let p = *slot as i32;
    *slot = (p + (((bit << 16) - p) >> RATE_STRIDE)) as u16;
}

// Sparse "stride" models capture fixed-period structure in binary media:
// 16-bit / stereo audio (stride 2, 4), RGB / RGBA images (stride 3, 4), and
// many fixed-record game formats. Each predicts the current byte from the
// same lane of previous samples. The mixer learns to trust them on media and
// ignore them on text, so they are safe to always include.
const STRIDES: [usize; 4] = [2, 3, 4, 8];
const NSTRIDE: usize = STRIDES.len();

// Mixer inputs:
//   7 counter models * 2 rates           = 14
// + NSTRIDE single-rate stride models
// + 1 match model
// + 1 bias
const NCTX: usize = 7;
const NIN: usize = NCTX * 2 + NSTRIDE + 2;
const STRIDE_IN: usize = NCTX * 2; // first stride input index
const MATCH_IN: usize = NCTX * 2 + NSTRIDE; // match input index
const BIAS_IN: usize = NIN - 1;

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

/// A dual-rate bit counter: a fast and a slow 16-bit `P(bit==1)` estimate.
#[derive(Clone, Copy)]
struct DualCounter {
    fast: u16,
    slow: u16,
}

impl DualCounter {
    const INIT: DualCounter = DualCounter {
        fast: 32768,
        slow: 32768,
    };

    #[inline]
    fn update(&mut self, bit: i32) {
        let f = self.fast as i32;
        self.fast = (f + (((bit << 16) - f) >> RATE_FAST)) as u16;
        let s = self.slow as i32;
        self.slow = (s + (((bit << 16) - s) >> RATE_SLOW)) as u16;
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
    // Context-model dual counters.
    t0: Vec<DualCounter>,        // order-0: partial byte (256)
    t1: Vec<DualCounter>,        // order-1: prev1<<8 | c0 (65536)
    th: [Vec<DualCounter>; 5],   // hashed: order-2,3,4,6, word

    // Rolling byte history (hist[0] = most recent).
    hist: [u8; 6],
    word_hash: u32,

    // Sparse stride models (single-rate u16 counters). Small, cache-resident
    // tables: stride contexts are low-cardinality (one or two sample bytes), so
    // a big table would only add cache misses without improving ratio.
    ts: [Vec<u16>; NSTRIDE],
    stride_mask: u32,
    stride_base: [u32; NSTRIDE],
    stride_idx: [usize; NSTRIDE],

    // Per-byte base hashes for the 5 hashed models.
    hbase: [u32; 5],
    // Slot indices chosen for the current bit (for the update step).
    idx: [usize; NCTX],

    // Partial byte: starts at 1, accumulates coded bits.
    c0: u32,

    hmask: u32,

    // Match model.
    buf: Vec<u8>,
    match_table: Vec<u32>,
    match_mask: u32,
    match_ptr: usize,
    match_len: u32,
    match_byte: i32, // predicted next byte, or -1 if no active match

    // Two-context mixer.
    wa: Vec<i32>, // [256][NIN] selected by previous byte
    wb: Vec<i32>, // [64][NIN]  selected by match-length bucket
    tx: [i32; NIN],
    ctx_a: usize,
    ctx_b: usize,
    pr: i32,

    // SSE chain.
    apm1: Apm,
    apm2: Apm,
    final_pr: i32,
}

impl Predictor {
    pub fn new(n: usize) -> Self {
        let _ = squash(0);
        let _ = stretch(2048);
        let hbits = table_bits(n);
        let hsize = 1usize << hbits;
        // Stride tables are capped smaller than the main tables: big enough to
        // avoid heavy collisions on 2-sample contexts, small enough to stay
        // cache-friendly on text where these models are mostly dead weight.
        let ssize = hsize.min(1 << 20);
        // The match table can be a touch larger; it stores one u32 per slot.
        let mbits = table_bits(n).min(HBITS_MAX);
        let msize = 1usize << mbits;
        Self {
            t0: vec![DualCounter::INIT; 256],
            t1: vec![DualCounter::INIT; 1 << 16],
            th: [
                vec![DualCounter::INIT; hsize],
                vec![DualCounter::INIT; hsize],
                vec![DualCounter::INIT; hsize],
                vec![DualCounter::INIT; hsize],
                vec![DualCounter::INIT; hsize],
            ],
            ts: [
                vec![32768u16; ssize],
                vec![32768u16; ssize],
                vec![32768u16; ssize],
                vec![32768u16; ssize],
            ],
            stride_mask: (ssize as u32) - 1,
            stride_base: [0; NSTRIDE],
            stride_idx: [0; NSTRIDE],
            hist: [0; 6],
            word_hash: 0,
            hbase: [0; 5],
            idx: [0; NCTX],
            c0: 1,
            hmask: (hsize as u32) - 1,
            buf: Vec::with_capacity(n),
            match_table: vec![MATCH_EMPTY; msize],
            match_mask: (msize as u32) - 1,
            match_ptr: 0,
            match_len: 0,
            match_byte: -1,
            wa: vec![0i32; 256 * NIN],
            wb: vec![0i32; 64 * NIN],
            tx: [0; NIN],
            ctx_a: 0,
            ctx_b: 0,
            pr: 2048,
            apm1: Apm::new(256),
            apm2: Apm::new(1024),
            final_pr: 2048,
        }
    }

    #[inline]
    pub fn predict(&mut self) -> i32 {
        let c0 = self.c0;

        // --- counter models: feed fast+slow stretched estimates ----------
        self.idx[0] = c0 as usize;
        let c = self.t0[self.idx[0]];
        self.tx[0] = stretch((c.fast >> 4) as i32);
        self.tx[1] = stretch((c.slow >> 4) as i32);

        let i1 = (((self.hist[0] as usize) << 8) | (c0 as usize & 0xff)) & 0xffff;
        self.idx[1] = i1;
        let c = self.t1[i1];
        self.tx[2] = stretch((c.fast >> 4) as i32);
        self.tx[3] = stretch((c.slow >> 4) as i32);

        let cmul = c0.wrapping_mul(PR2);
        for k in 0..5 {
            let slot = ((self.hbase[k] ^ cmul) & self.hmask) as usize;
            self.idx[k + 2] = slot;
            let c = self.th[k][slot];
            self.tx[4 + k * 2] = stretch((c.fast >> 4) as i32);
            self.tx[5 + k * 2] = stretch((c.slow >> 4) as i32);
        }

        // --- sparse stride models ---------------------------------------
        for k in 0..NSTRIDE {
            let slot = ((self.stride_base[k] ^ cmul) & self.stride_mask) as usize;
            self.stride_idx[k] = slot;
            self.tx[STRIDE_IN + k] = stretch((self.ts[k][slot] >> 4) as i32);
        }

        // --- match model -------------------------------------------------
        self.tx[MATCH_IN] = self.match_prediction(c0);

        // --- bias --------------------------------------------------------
        self.tx[BIAS_IN] = 256;

        // --- two-context logistic mixing ---------------------------------
        self.ctx_a = self.hist[0] as usize;
        self.ctx_b = (self.match_len.min(63)) as usize;
        let da = self.dot(&self.wa, self.ctx_a);
        let db = self.dot(&self.wb, self.ctx_b);
        self.pr = squash((da + db) >> 1);

        // --- SSE refinement ---------------------------------------------
        let p1 = self.apm1.refine(self.pr, self.ctx_a);
        let mut p = (self.pr + p1 * 3) >> 2;
        let p2 = self.apm2.refine(p, (self.hbase[1] & 0x3ff) as usize);
        p = (p + p2 * 3) >> 2;
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
        self.t0[self.idx[0]].update(bit);
        self.t1[self.idx[1]].update(bit);
        for k in 0..5 {
            self.th[k][self.idx[k + 2]].update(bit);
        }
        for k in 0..NSTRIDE {
            counter_update(&mut self.ts[k][self.stride_idx[k]], bit);
        }

        // Mixer weights: gradient step on coding error for both views.
        let err = ((bit << 12) - self.pr) * 7;
        Self::train(&mut self.wa, self.ctx_a, &self.tx, err);
        Self::train(&mut self.wb, self.ctx_b, &self.tx, err);

        self.apm1.update(bit);
        self.apm2.update(bit);

        self.c0 = (self.c0 << 1) | (bit as u32);
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

        // Refresh / seed a match from the suffix hash. On a fresh seed we
        // *verify* the candidate by extending backward, both to reject hash
        // collisions and to recover the true match length — long verified
        // matches let the mixer predict the continuation near-certainly, which
        // is what captures long-range / cross-copy redundancy in big archives.
        if pos >= MATCH_MIN {
            let h = (self.suffix_hash() & self.match_mask) as usize;
            let cand = self.match_table[h];
            self.match_table[h] = pos as u32;
            if self.match_len == 0 && cand != MATCH_EMPTY && (cand as usize) < pos {
                let c = cand as usize;
                let max = c.min(pos);
                let mut l = 0usize;
                while l < max && self.buf[c - 1 - l] == self.buf[pos - 1 - l] && l < 0xffff {
                    l += 1;
                }
                if l >= MATCH_MIN {
                    self.match_ptr = c;
                    self.match_len = l as u32;
                }
            }
        }
        self.match_byte = if self.match_len > 0 && self.match_ptr < self.buf.len() {
            self.buf[self.match_ptr] as i32
        } else {
            -1
        };

        // --- context history --------------------------------------------
        self.hist.copy_within(0..5, 1);
        self.hist[0] = byte;
        self.c0 = 1;

        if byte.is_ascii_alphabetic() {
            let lower = byte | 0x20;
            self.word_hash = self
                .word_hash
                .wrapping_add(lower as u32 + 1)
                .wrapping_mul(PR1);
        } else {
            self.word_hash = 0;
        }

        self.hbase[0] = hash_ctx(&self.hist[0..2], 2);
        self.hbase[1] = hash_ctx(&self.hist[0..3], 3);
        self.hbase[2] = hash_ctx(&self.hist[0..4], 4);
        self.hbase[3] = hash_ctx(&self.hist[0..6], 6);
        self.hbase[4] = self.word_hash.wrapping_mul(PR1) ^ 0xABCD_1234;

        // Stride bases: predict the upcoming byte (at index buf.len()) from the
        // same lane of the previous one/two samples `stride` bytes back.
        let n = self.buf.len();
        for (k, &s) in STRIDES.iter().enumerate() {
            let b1 = if n >= s { self.buf[n - s] as u32 } else { 0 };
            let b2 = if n >= 2 * s { self.buf[n - 2 * s] as u32 } else { 0 };
            let mut h = (s as u32).wrapping_mul(PR1).wrapping_add(0x55AA_33CC);
            h = (h ^ (b1 + 1)).wrapping_mul(PR1);
            h = (h ^ (b2 + 1)).wrapping_mul(PR1);
            self.stride_base[k] = h ^ (h >> 15);
        }
    }

    /// Hash of the last `MATCH_MIN` committed bytes.
    #[inline]
    fn suffix_hash(&self) -> u32 {
        let n = self.buf.len();
        let mut h: u32 = 0x811C_9DC5;
        for &b in &self.buf[n - MATCH_MIN..n] {
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
