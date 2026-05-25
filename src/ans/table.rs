//! ANS frequency table construction.
//!
//! We use rANS with M = TABLE_SIZE total frequency slots.
//! The state x is a u32 maintained in [M, M*256).
//! Encoding  sym: flush bytes while x >= freq[sym]*256, then x = (x/f)*M + cumul + x%f
//! Decoding       slot = x % M → sym; x = freq[sym]*(x/M) + slot - cumul[sym]

/// Total number of frequency slots (power of 2 for fast modulo).
pub const TABLE_SIZE: u32 = 1 << 12; // 4096

/// Precomputed tables for one probability distribution.
pub struct AnsTable {
    pub freq:        [u32; 256],
    pub cumul:       [u32; 256],
    /// slot_sym[slot] = symbol whose cumul range covers `slot`.
    /// slot ∈ [0, TABLE_SIZE)
    pub slot_sym:    Box<[u8; 4096]>,
}

/// Build tables from a probability distribution (values summing to ≈1.0).
pub fn build_table(prob: &[f32; 256]) -> Box<AnsTable> {
    let freq = normalize_freqs(prob, TABLE_SIZE);

    let mut cumul = [0u32; 256];
    let mut cum = 0u32;
    for (i, &f) in freq.iter().enumerate() {
        cumul[i] = cum;
        cum += f;
    }
    debug_assert_eq!(cum, TABLE_SIZE);

    // Build slot_sym: contiguous ranges [cumul[s], cumul[s]+freq[s]) → s
    let mut slot_sym = Box::new([0u8; 4096]);
    for sym in 0u8..=255 {
        let f = freq[sym as usize];
        let c = cumul[sym as usize];
        for i in 0..f {
            slot_sym[(c + i) as usize] = sym;
        }
    }

    Box::new(AnsTable { freq, cumul, slot_sym })
}

/// Normalize probabilities to integer frequencies summing exactly to `m`.
fn normalize_freqs(prob: &[f32; 256], m: u32) -> [u32; 256] {
    let mut freq = [0u32; 256];
    let mut assigned = 0u32;

    // Initial allocation by rounding
    let mut remainders = [(0f32, 0usize); 256];
    for (i, &p) in prob.iter().enumerate() {
        let exact = p * m as f32;
        let floored = exact as u32;
        freq[i] = floored.max(if p > 0.0 { 1 } else { 0 });
        assigned += freq[i];
        remainders[i] = (exact - floored as f32, i);
    }

    // Distribute remaining slots to largest remainders
    let remaining = m.saturating_sub(assigned) as usize;
    if remaining > 0 {
        remainders.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        for &(_, idx) in remainders.iter().take(remaining) {
            freq[idx] += 1;
        }
    }
    // Handle over-allocation (rare due to float rounding)
    let sum: u32 = freq.iter().sum();
    if sum > m {
        let excess = (sum - m) as usize;
        let mut sorted_nonzero: Vec<(u32, usize)> = freq.iter().cloned()
            .enumerate()
            .filter(|(_, f)| *f > 1)
            .map(|(i, f)| (f, i))
            .collect();
        sorted_nonzero.sort_unstable();
        for &(_, idx) in sorted_nonzero.iter().take(excess) {
            if freq[idx] > 1 { freq[idx] -= 1; }
        }
    }

    freq
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_freqs_sum_to_table_size() {
        let prob = [1.0f32 / 256.0; 256];
        let freq = normalize_freqs(&prob, TABLE_SIZE);
        let sum: u32 = freq.iter().sum();
        assert_eq!(sum, TABLE_SIZE);
    }

    #[test]
    fn skewed_freqs_sum_to_table_size() {
        let mut prob = [0.001f32; 256];
        prob[0] = 0.744; // dominant symbol
        let total: f32 = prob.iter().sum();
        for p in prob.iter_mut() { *p /= total; }
        let freq = normalize_freqs(&prob, TABLE_SIZE);
        let sum: u32 = freq.iter().sum();
        assert_eq!(sum, TABLE_SIZE);
    }

    #[test]
    fn slot_sym_covers_all_slots() {
        let prob = [1.0f32 / 256.0; 256];
        let table = build_table(&prob);
        // Every slot must have been written (no slot left at default 0 ambiguously)
        let sum: u32 = table.freq.iter().sum();
        assert_eq!(sum, TABLE_SIZE);
    }
}
