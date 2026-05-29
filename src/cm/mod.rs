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
//! Encoder and decoder run the identical model in lockstep, so the model is
//! never stored. Because both sides execute the same deterministic code, hash
//! collisions and SSE quirks can only cost ratio, never correctness.

mod coder;
mod predictor;

use coder::{Decoder, Encoder};
use predictor::Predictor;

/// Compress `data` into a self-contained CM2 payload.
///
/// The payload carries no length prefix — the caller is expected to know how
/// many bytes to decode (CPGC stores `orig_len` in its container header).
pub fn encode(data: &[u8]) -> Vec<u8> {
    if data.is_empty() {
        return Vec::new();
    }
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

/// Decode exactly `n` bytes from a CM2 payload produced by [`encode`].
pub fn decode(payload: &[u8], n: usize) -> Vec<u8> {
    if n == 0 {
        return Vec::new();
    }
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
