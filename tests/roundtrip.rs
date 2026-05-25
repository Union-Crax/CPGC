//! Round-trip correctness tests.

use cpgc::ans::encode::AnsEncoder;
use cpgc::ans::decode::AnsDecoder;

/// Uniform distribution over all 256 symbols.
fn uniform_prob() -> [f32; 256] { [1.0f32 / 256.0; 256] }

/// Skewed distribution: symbol 0 has prob 0.5, rest share 0.5.
fn skewed_prob() -> [f32; 256] {
    let mut p = [0.5f32 / 255.0; 256];
    p[0] = 0.5;
    p
}

fn ans_roundtrip(symbols: &[u8], prob: &[f32; 256]) {
    // Encode
    let mut enc = AnsEncoder::new(prob);
    for &s in symbols { enc.encode(s); }
    let compressed = enc.finish();

    // Decode
    let mut dec = AnsDecoder::new(&compressed, prob).expect("decoder init failed");
    let mut decoded = Vec::with_capacity(symbols.len());
    for _ in 0..symbols.len() {
        decoded.push(dec.decode().expect("decode returned None"));
    }

    assert_eq!(decoded, symbols,
        "roundtrip mismatch (compressed {} bytes for {} symbols)",
        compressed.len(), symbols.len());
}

#[test]
fn ans_roundtrip_uniform_single() {
    ans_roundtrip(&[42u8], &uniform_prob());
}

#[test]
fn ans_roundtrip_uniform_short() {
    let syms: Vec<u8> = (0u8..=15).collect();
    ans_roundtrip(&syms, &uniform_prob());
}

#[test]
fn ans_roundtrip_uniform_long() {
    // 1000 symbols, cycling through all byte values
    let syms: Vec<u8> = (0u16..1000).map(|i| (i % 256) as u8).collect();
    ans_roundtrip(&syms, &uniform_prob());
}

#[test]
fn ans_roundtrip_skewed() {
    // 200 bytes of mostly symbol 0
    let mut syms = vec![0u8; 180];
    syms.extend(1u8..=20);
    ans_roundtrip(&syms, &skewed_prob());
}

#[test]
fn ans_roundtrip_all_same() {
    let syms = vec![b'A'; 100];
    // Laplace-smoothed distribution so all symbols have non-zero frequency
    let p_smoothed: [f32; 256] = {
        let mut pp = [0.001f32; 256];
        pp[b'A' as usize] = 1.0 - 0.001 * 255.0;
        let s: f32 = pp.iter().sum();
        pp.iter().map(|v| v / s).collect::<Vec<_>>().try_into().unwrap()
    };
    ans_roundtrip(&syms, &p_smoothed);
}

#[test]
fn ans_compression_ratio_skewed() {
    // With a strongly skewed distribution, compressed size << raw size
    let syms = vec![0u8; 500]; // 500 bytes of the dominant symbol
    let mut enc = AnsEncoder::new(&skewed_prob());
    for &s in &syms { enc.encode(s); }
    let compressed = enc.finish();
    // P(0) = 0.5, so each symbol costs ~1 bit → 500 symbols ≈ 500 bits ≈ 63 bytes + 4 header
    // Allow generous upper bound: < 150 bytes (raw = 500)
    assert!(
        compressed.len() < 150,
        "expected < 150 bytes for 500 dominant symbols, got {}",
        compressed.len()
    );
}
