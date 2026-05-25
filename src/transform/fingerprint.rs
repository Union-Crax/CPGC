//! Chunk fingerprinting for fast transform candidate filtering.

/// FNV-1a 64-bit hash of a byte slice.
pub fn fingerprint(data: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}
