//! Shannon entropy estimation (float arithmetic, per byte).

/// Returns bits per byte (0.0 = all same, 8.0 = uniform random).
pub fn entropy_bits(data: &[u8]) -> f32 {
    if data.is_empty() { return 0.0; }
    let mut counts = [0u32; 256];
    for &b in data { counts[b as usize] += 1; }
    let n = data.len() as f32;
    counts.iter()
        .filter(|&&c| c > 0)
        .map(|&c| {
            let p = c as f32 / n;
            -p * p.log2()
        })
        .sum()
}

/// Returns true if data looks already-compressed or encrypted (entropy > 7.5 bits/byte).
pub fn is_high_entropy(data: &[u8]) -> bool {
    entropy_bits(data) > 7.5
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_is_8_bits() {
        let data: Vec<u8> = (0u8..=255).collect();
        let e = entropy_bits(&data);
        assert!((e - 8.0).abs() < 0.01, "entropy = {}", e);
    }

    #[test]
    fn constant_is_0_bits() {
        let data = vec![0u8; 1000];
        assert_eq!(entropy_bits(&data), 0.0);
    }
}
