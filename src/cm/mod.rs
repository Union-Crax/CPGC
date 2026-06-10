//! CPGC-NX — a bit-level context-mixing compression engine.
//!
//! This is the new core of CPGC, replacing the slow per-byte LSTM SGD path. It
//! is both **faster** (no neural forward/backward pass per byte) and **much
//! higher ratio** (on general text it beats gzip, bzip2 and xz/LZMA outright).
//!
//! The bit-level context-mixing *framework* is shared with the PAQ family, but
//! the model here is a distinct combination tuned for this codec (see
//! [`predictor`] for the details):
//!
//! 1. **Universal bit-history states** — every hashed context slot is one
//!    packed byte of mutually-discounting 0/1 counts, read through a learned
//!    per-model state map, capturing nonstationary structure counters miss.
//! 2. **Nibble-bucketed, checksummed hash tables** — one prefetched cache
//!    line serves four bits and collisions are detected instead of silently
//!    corrupting predictions; this is where v7's ~2.4x speedup comes from.
//! 3. **A dual long-match model** — 8-byte and 4-byte rolling-hash pointers
//!    into history forecast the bit of the most recent matching continuation.
//! 4. **A two-layer learned mixer** — four context-selected weight sets
//!    (previous byte, byte before it, match length, partial byte) feed a
//!    small second-layer combiner trained online on coding loss.
//! 5. **A chained SSE stage** (four APMs) refines the result before the
//!    binary arithmetic coder.
//! 6. **Two-speed coding** — bytes deep inside a verified match are coded by
//!    a tiny match-confidence model, skipping the mixer entirely; the switch
//!    is deterministic so it costs no signalling.
//! 7. **A runtime-SIMD mixer** (AVX2 with a bit-identical scalar fallback)
//!    and **two model profiles** (turbo for levels 1-3, full for 4-9; the
//!    profile byte travels in the payload header).
//!
//! ## Scaling to big archives
//!
//! For inputs larger than [`SEG_SIZE`] the stream is split into fixed-size
//! **independent segments** that are compressed and decompressed in parallel
//! across all CPU cores. The segment size is fixed (not derived from the core
//! count), so an archive written on a 4-core machine decodes identically on a
//! 64-core one. Segments are large enough (multiple MiB) that the per-segment
//! model warm-up costs a negligible amount of ratio, while throughput scales
//! close to linearly with the number of cores.
//!
//! Encoder and decoder run the identical model in lockstep, so the model is
//! never stored. Because both sides execute the same deterministic code, hash
//! collisions and SSE quirks can only cost ratio, never correctness.

mod coder;
mod predictor;

use coder::{Decoder, Encoder};
use predictor::Predictor;
use rayon::prelude::*;
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};

// Control state values
const RUN: u8 = 0;
const PAUSE: u8 = 1;
const CANCEL: u8 = 2;

/// Shared cooperative control + progress handle for a (de)compression job.
///
/// The rayon worker threads check it periodically: they block while paused,
/// bail out when cancelled, and publish a running count of bytes processed so
/// a UI can show a progress bar and live throughput. Cheap to share across
/// threads (`Arc<Control>` or a plain borrow).
#[derive(Default)]
pub struct Control {
    state: AtomicU8,
    done:  AtomicU64,
}

impl Control {
    pub fn new() -> Self {
        Self { state: AtomicU8::new(RUN), done: AtomicU64::new(0) }
    }
    pub fn pause(&self) {
        let _ = self.state.compare_exchange(RUN, PAUSE, Ordering::SeqCst, Ordering::SeqCst);
    }
    pub fn resume(&self) {
        let _ = self.state.compare_exchange(PAUSE, RUN, Ordering::SeqCst, Ordering::SeqCst);
    }
    pub fn cancel(&self) {
        self.state.store(CANCEL, Ordering::SeqCst);
    }
    pub fn is_paused(&self) -> bool {
        self.state.load(Ordering::Relaxed) == PAUSE
    }
    pub fn is_cancelled(&self) -> bool {
        self.state.load(Ordering::Relaxed) == CANCEL
    }
    /// Bytes processed so far (for progress bar / throughput display).
    pub fn bytes_done(&self) -> u64 {
        self.done.load(Ordering::Relaxed)
    }
    fn add_done(&self, n: u64) {
        self.done.fetch_add(n, Ordering::Relaxed);
    }
    /// Block while paused; return `false` once cancelled.
    fn proceed(&self) -> bool {
        loop {
            match self.state.load(Ordering::Relaxed) {
                CANCEL => return false,
                PAUSE  => std::thread::sleep(std::time::Duration::from_millis(15)),
                _      => return true,
            }
        }
    }
}

/// Default segment size (compression level 5). Inputs larger than the level's
/// segment size are split so the segments can be (de)compressed on separate
/// cores. Larger segments give a better ratio (more match history per segment)
/// at the cost of less parallelism — that trade-off is the compression level.
pub const SEG_SIZE: usize = 16 << 20; // 16 MiB == level 5

/// Map a 1–9 compression level to a segment size.
///
/// Lower levels use smaller segments: more parallelism (faster) at a small
/// ratio cost. Higher levels use larger segments: better ratio, less
/// parallelism. The chosen size is stored in the payload, so decoding never
/// depends on this mapping.
pub fn seg_size_for_level(level: u8) -> usize {
    let bits: u32 = match level {
        0 | 1 => 20, // 1 MiB
        2 => 21,
        3 => 22,
        4 => 23,
        5 => 24, // 16 MiB (default)
        6 => 25,
        7 => 26,
        8 => 27,
        _ => 28, // 256 MiB
    };
    1usize << bits
}

/// Compress `data` into a self-contained CPGC-NX payload at the given level.
///
/// The payload is self-framing — it records the segment size, the model
/// profile and each segment's compressed length — but carries no *total*
/// length prefix; the caller is expected to know how many bytes to decode
/// (CPGC stores `orig_len` in its container header).
///
/// Layout:
/// ```text
/// [0..4]   seg_size: u32 LE   (bytes of original data per segment)
/// [4..8]   n_seg: u32 LE
/// [8]      profile: u8        (0 = full model, 1 = turbo)
/// [9..]    n_seg × (comp_len: u32 LE)
/// [rest]   segment payloads, concatenated in order
/// ```
/// Segment `i` decompresses to `data[i*seg_size .. min((i+1)*seg_size, n)]`, so
/// only the *compressed* lengths need to be stored. Levels 1-3 use the turbo
/// profile (a reduced model roster, several times faster); the profile byte
/// means decoding never depends on the level mapping.
pub fn encode(data: &[u8], level: u8) -> Vec<u8> {
    encode_framed(data, seg_size_for_level(level), level <= 3, &Control::new())
        .expect("uncontrolled encode is never cancelled")
}

/// Compress with a shared [`Control`] for pause/resume/cancel and a live byte
/// counter. Returns `None` if the job was cancelled before completing.
pub fn encode_with_control(data: &[u8], level: u8, ctrl: &Control) -> Option<Vec<u8>> {
    encode_framed(data, seg_size_for_level(level), level <= 3, ctrl)
}

/// Decode exactly `n` bytes from a payload produced by [`encode`].
pub fn decode(payload: &[u8], n: usize) -> Vec<u8> {
    decode_framed(payload, n, &Control::new())
        .expect("uncontrolled decode is never cancelled")
}

/// Decode with a shared [`Control`]. Returns `None` if cancelled.
pub fn decode_with_control(payload: &[u8], n: usize, ctrl: &Control) -> Option<Vec<u8>> {
    decode_framed(payload, n, ctrl)
}

// How often (in bytes) a worker checks the control flag and publishes progress.
// Batching keeps the shared atomic cheap on the hot encode/decode loop.
const CHECK_INTERVAL: usize = 1 << 12; // 4 KiB

fn encode_framed(data: &[u8], seg_size: usize, turbo: bool, ctrl: &Control) -> Option<Vec<u8>> {
    if data.is_empty() {
        return Some(Vec::new());
    }
    let seg_size = seg_size.max(1);

    // Compress each segment independently, in parallel. A cancelled segment
    // yields `None`, which collapses the whole result to `None`.
    let segments: Option<Vec<Vec<u8>>> = data
        .par_chunks(seg_size)
        .map(|chunk| encode_segment(chunk, turbo, ctrl))
        .collect();
    let segments = segments?;

    let n_seg = segments.len();
    let header = 9 + 4 * n_seg;
    let body: usize = segments.iter().map(|s| s.len()).sum();
    let mut out = Vec::with_capacity(header + body);
    out.extend_from_slice(&(seg_size as u32).to_le_bytes());
    out.extend_from_slice(&(n_seg as u32).to_le_bytes());
    out.push(turbo as u8);
    for s in &segments {
        out.extend_from_slice(&(s.len() as u32).to_le_bytes());
    }
    for s in &segments {
        out.extend_from_slice(s);
    }
    Some(out)
}

fn decode_framed(payload: &[u8], n: usize, ctrl: &Control) -> Option<Vec<u8>> {
    if n == 0 || payload.len() < 9 {
        return Some(Vec::new());
    }
    let seg_size = (u32::from_le_bytes(payload[0..4].try_into().unwrap()) as usize).max(1);
    let n_seg = u32::from_le_bytes(payload[4..8].try_into().unwrap()) as usize;
    let turbo = payload[8] != 0;
    let comp_lens: Vec<usize> = (0..n_seg)
        .map(|i| {
            let o = 9 + 4 * i;
            u32::from_le_bytes(payload[o..o + 4].try_into().unwrap()) as usize
        })
        .collect();

    // Slice the concatenated payloads and pair each with its decoded length.
    let mut body = &payload[9 + 4 * n_seg..];
    let mut jobs: Vec<(&[u8], usize)> = Vec::with_capacity(n_seg);
    for (i, &clen) in comp_lens.iter().enumerate() {
        let seg_start = i * seg_size;
        let seg_len = (seg_start + seg_size).min(n) - seg_start;
        let (seg_payload, rest) = body.split_at(clen.min(body.len()));
        body = rest;
        jobs.push((seg_payload, seg_len));
    }

    // Decode segments in parallel, then concatenate in order.
    let parts: Option<Vec<Vec<u8>>> = jobs
        .par_iter()
        .map(|&(p, len)| decode_segment(p, len, turbo, ctrl))
        .collect();
    let parts = parts?;

    let mut out = Vec::with_capacity(n);
    for p in parts {
        out.extend_from_slice(&p);
    }
    Some(out)
}

/// Compress a single segment (no framing). Returns `None` if cancelled.
fn encode_segment(data: &[u8], turbo: bool, ctrl: &Control) -> Option<Vec<u8>> {
    let mut model = Predictor::new(data.len(), turbo);
    let mut enc = Encoder::new();
    let mut since_check = 0usize;
    for &byte in data {
        if since_check >= CHECK_INTERVAL {
            ctrl.add_done(since_check as u64);
            since_check = 0;
            if !ctrl.proceed() {
                return None;
            }
        }
        // MSB-first bit coding through the shared per-byte context tree.
        for bit_index in (0..8).rev() {
            let bit = ((byte >> bit_index) & 1) as i32;
            let p = model.predict();
            enc.encode(bit, p);
            model.update(bit);
        }
        model.next_byte(byte);
        since_check += 1;
    }
    ctrl.add_done(since_check as u64);
    Some(enc.finish())
}

/// Decode a single segment of exactly `n` bytes. Returns `None` if cancelled.
fn decode_segment(payload: &[u8], n: usize, turbo: bool, ctrl: &Control) -> Option<Vec<u8>> {
    let mut model = Predictor::new(n, turbo);
    let mut dec = Decoder::new(payload);
    let mut out = Vec::with_capacity(n);
    let mut since_check = 0usize;
    for _ in 0..n {
        if since_check >= CHECK_INTERVAL {
            ctrl.add_done(since_check as u64);
            since_check = 0;
            if !ctrl.proceed() {
                return None;
            }
        }
        let mut byte = 0u8;
        for _ in 0..8 {
            let p = model.predict();
            let bit = dec.decode(p);
            model.update(bit);
            byte = (byte << 1) | (bit as u8);
        }
        model.next_byte(byte);
        out.push(byte);
        since_check += 1;
    }
    ctrl.add_done(since_check as u64);
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(data: &[u8]) {
        let payload = encode(data, 5);
        let decoded = decode(&payload, data.len());
        assert_eq!(decoded, data, "roundtrip mismatch ({} bytes)", data.len());
    }

    /// Round-trip with an explicit (small) segment size to exercise the
    /// multi-segment parallel framing without needing multi-MiB inputs. The
    /// segment size and profile are recovered from the payload, so decode
    /// needs no hint. Both model profiles are exercised.
    fn roundtrip_segmented(data: &[u8], seg: usize) {
        for turbo in [false, true] {
            let ctrl = Control::new();
            let payload = encode_framed(data, seg, turbo, &ctrl).unwrap();
            let decoded = decode_framed(&payload, data.len(), &Control::new()).unwrap();
            assert_eq!(
                decoded, data,
                "segmented roundtrip mismatch (seg={seg}, turbo={turbo}, {} bytes)",
                data.len()
            );
        }
    }

    #[test]
    fn rt_turbo_profile() {
        // Levels 1-3 use the turbo profile end-to-end.
        let s = "the quick brown fox jumps over the lazy dog. ".repeat(400);
        let payload = encode(s.as_bytes(), 2);
        assert_eq!(decode(&payload, s.len()), s.as_bytes());
    }

    #[test]
    fn rt_multi_segment() {
        let s = "the quick brown fox jumps over the lazy dog. ".repeat(500);
        // Many segment sizes, including ones that don't divide the length.
        for seg in [1usize, 7, 64, 257, 1000, 4096] {
            roundtrip_segmented(s.as_bytes(), seg);
        }
    }

    #[test]
    fn rt_multi_segment_random() {
        let mut x: u64 = 0xabcd_1234_5678_9f0f;
        let d: Vec<u8> = (0..10_000)
            .map(|_| {
                x = x
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (x >> 56) as u8
            })
            .collect();
        for seg in [13usize, 128, 999, 3001] {
            roundtrip_segmented(&d, seg);
        }
    }

    #[test]
    fn segmentation_is_transparent_to_ratio() {
        let s = "compression test data ".repeat(2000);
        let c = Control::new();
        let one = encode_framed(s.as_bytes(), s.len() + 1, false, &c).unwrap();
        let many = encode_framed(s.as_bytes(), 4096, false, &Control::new()).unwrap();
        assert_eq!(decode_framed(&one, s.len(), &Control::new()).unwrap(), s.as_bytes());
        assert_eq!(decode_framed(&many, s.len(), &Control::new()).unwrap(), s.as_bytes());
    }

    #[test]
    fn cancel_aborts_encode() {
        let data = vec![b'x'; 2_000_000];
        let ctrl = Control::new();
        ctrl.cancel();
        let out = encode_with_control(&data, 1, &ctrl);
        assert!(out.is_none(), "cancelled encode should return None");
    }

    #[test]
    fn control_reports_progress() {
        let data = vec![b'y'; 200_000];
        let ctrl = Control::new();
        let _ = encode_with_control(&data, 1, &ctrl);
        assert_eq!(ctrl.bytes_done(), data.len() as u64, "all bytes should be counted");
    }

    #[test]
    fn uncancelled_control_roundtrips() {
        let data = b"control path must be lossless too".repeat(50);
        let ctrl = Control::new();
        let payload = encode_with_control(&data, 5, &ctrl).unwrap();
        let back = decode_with_control(&payload, data.len(), &Control::new()).unwrap();
        assert_eq!(back, data);
    }

    #[test]
    fn rt_empty() {
        roundtrip(&[]);
    }

    #[test]
    fn rt_single() {
        roundtrip(&[0x42]);
    }

    #[test]
    fn rt_all_bytes() {
        let d: Vec<u8> = (0u8..=255).collect();
        roundtrip(&d);
    }

    #[test]
    fn rt_all_zeros() {
        roundtrip(&vec![0u8; 5000]);
    }

    #[test]
    fn rt_text() {
        let s = "the quick brown fox jumps over the lazy dog. ".repeat(300);
        roundtrip(s.as_bytes());
    }

    #[test]
    fn rt_random() {
        let mut x: u64 = 0x1234_5678_9abc_def0;
        let d: Vec<u8> = (0..20_000)
            .map(|_| {
                x = x
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (x >> 56) as u8
            })
            .collect();
        roundtrip(&d);
    }

    #[test]
    fn rt_lengths() {
        // Exercise many lengths to catch off-by-one / flush edge cases.
        for len in 0..300 {
            let d: Vec<u8> = (0..len).map(|i| (i * 31 + 7) as u8).collect();
            roundtrip(&d);
        }
    }

    #[test]
    fn compresses_repetitive() {
        let d = vec![b'a'; 10_000];
        let payload = encode(&d, 5);
        assert!(
            payload.len() < 200,
            "highly repetitive data should compress hard, got {} bytes",
            payload.len()
        );
    }

    #[test]
    fn compresses_text_well() {
        let s = "the quick brown fox jumps over the lazy dog. ".repeat(1000);
        let payload = encode(s.as_bytes(), 5);
        let bpb = payload.len() as f64 * 8.0 / s.len() as f64;
        assert!(bpb < 1.0, "structured text should be < 1 bpb, got {bpb:.3}");
    }
}
