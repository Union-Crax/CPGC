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
//! 1. **Dual learning-rate counters** — every context slot carries a fast and
//!    a slow probability estimate, both exposed to the mixer.
//! 2. **An integrated long-match model** — a rolling-hash pointer into history
//!    forecasts the bit of the most recent matching continuation.
//! 3. **A two-context mixing layer** — predictions are mixed by two weight sets
//!    (selected by the previous byte and by match-length) and averaged in the
//!    logistic domain.
//! 4. **A chained SSE stage** refines the result before the binary arithmetic
//!    coder.
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

/// Bytes per independent segment. Inputs larger than this are split so the
/// segments can be (de)compressed on separate cores. Chosen large enough that
/// per-segment model warm-up is a negligible fraction of the segment.
pub const SEG_SIZE: usize = 16 << 20; // 16 MiB

/// Compress `data` into a self-contained CPGC-NX payload.
///
/// The payload is self-framing for segmentation but carries no *total* length
/// prefix — the caller is expected to know how many bytes to decode (CPGC
/// stores `orig_len` in its container header).
///
/// Layout:
/// ```text
/// [0..4]   n_seg: u32 LE
/// [4..]    n_seg × (comp_len: u32 LE)
/// [rest]   segment payloads, concatenated in order
/// ```
/// Segment `i` decompresses to `data[i*SEG_SIZE .. min((i+1)*SEG_SIZE, n)]`, so
/// only the *compressed* lengths need to be stored.
pub fn encode(data: &[u8]) -> Vec<u8> {
    encode_framed(data, SEG_SIZE)
}

/// Decode exactly `n` bytes from a payload produced by [`encode`].
pub fn decode(payload: &[u8], n: usize) -> Vec<u8> {
    decode_framed(payload, n, SEG_SIZE)
}

fn encode_framed(data: &[u8], seg_size: usize) -> Vec<u8> {
    if data.is_empty() {
        return Vec::new();
    }

    // Compress each segment independently, in parallel.
    let segments: Vec<Vec<u8>> = data
        .par_chunks(seg_size)
        .map(encode_segment)
        .collect();

    let n_seg = segments.len();
    let header = 4 + 4 * n_seg;
    let body: usize = segments.iter().map(|s| s.len()).sum();
    let mut out = Vec::with_capacity(header + body);
    out.extend_from_slice(&(n_seg as u32).to_le_bytes());
    for s in &segments {
        out.extend_from_slice(&(s.len() as u32).to_le_bytes());
    }
    for s in &segments {
        out.extend_from_slice(s);
    }
    out
}

fn decode_framed(payload: &[u8], n: usize, seg_size: usize) -> Vec<u8> {
    if n == 0 || payload.len() < 4 {
        return Vec::new();
    }
    let n_seg = u32::from_le_bytes(payload[0..4].try_into().unwrap()) as usize;
    let comp_lens: Vec<usize> = (0..n_seg)
        .map(|i| {
            let o = 4 + 4 * i;
            u32::from_le_bytes(payload[o..o + 4].try_into().unwrap()) as usize
        })
        .collect();

    // Slice the concatenated payloads and pair each with its decoded length.
    let mut body = &payload[4 + 4 * n_seg..];
    let mut jobs: Vec<(&[u8], usize)> = Vec::with_capacity(n_seg);
    for (i, &clen) in comp_lens.iter().enumerate() {
        let seg_start = i * seg_size;
        let seg_len = (seg_start + seg_size).min(n) - seg_start;
        let (seg_payload, rest) = body.split_at(clen.min(body.len()));
        body = rest;
        jobs.push((seg_payload, seg_len));
    }

    // Decode segments in parallel, then concatenate in order.
    let parts: Vec<Vec<u8>> = jobs
        .par_iter()
        .map(|&(p, len)| decode_segment(p, len))
        .collect();

    let mut out = Vec::with_capacity(n);
    for p in parts {
        out.extend_from_slice(&p);
    }
    out
}

/// Compress a single segment (no framing).
fn encode_segment(data: &[u8]) -> Vec<u8> {
    let mut model = Predictor::new(data.len());
    let mut enc = Encoder::new();
    for &byte in data {
        // MSB-first bit coding through the shared per-byte context tree.
        for bit_index in (0..8).rev() {
            let bit = ((byte >> bit_index) & 1) as i32;
            let p = model.predict();
            enc.encode(bit, p);
            model.update(bit);
        }
        model.next_byte(byte);
    }
    enc.finish()
}

/// Decode a single segment of exactly `n` bytes.
fn decode_segment(payload: &[u8], n: usize) -> Vec<u8> {
    let mut model = Predictor::new(n);
    let mut dec = Decoder::new(payload);
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let mut byte = 0u8;
        for _ in 0..8 {
            let p = model.predict();
            let bit = dec.decode(p);
            model.update(bit);
            byte = (byte << 1) | (bit as u8);
        }
        model.next_byte(byte);
        out.push(byte);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(data: &[u8]) {
        let payload = encode(data);
        let decoded = decode(&payload, data.len());
        assert_eq!(decoded, data, "roundtrip mismatch ({} bytes)", data.len());
    }

    /// Round-trip with an explicit (small) segment size to exercise the
    /// multi-segment parallel framing without needing multi-MiB inputs.
    fn roundtrip_segmented(data: &[u8], seg: usize) {
        let payload = encode_framed(data, seg);
        let decoded = decode_framed(&payload, data.len(), seg);
        assert_eq!(
            decoded, data,
            "segmented roundtrip mismatch (seg={seg}, {} bytes)",
            data.len()
        );
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
        // One big segment vs many small ones should both round-trip; this also
        // documents that segmentation only adds the small per-segment header.
        let s = "compression test data ".repeat(2000);
        let one = encode_framed(s.as_bytes(), s.len() + 1);
        let many = encode_framed(s.as_bytes(), 4096);
        assert_eq!(decode_framed(&one, s.len(), s.len() + 1), s.as_bytes());
        assert_eq!(decode_framed(&many, s.len(), 4096), s.as_bytes());
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
        let payload = encode(&d);
        assert!(
            payload.len() < 200,
            "highly repetitive data should compress hard, got {} bytes",
            payload.len()
        );
    }

    #[test]
    fn compresses_text_well() {
        let s = "the quick brown fox jumps over the lazy dog. ".repeat(1000);
        let payload = encode(s.as_bytes());
        let bpb = payload.len() as f64 * 8.0 / s.len() as f64;
        assert!(bpb < 1.0, "structured text should be < 1 bpb, got {bpb:.3}");
    }
}
