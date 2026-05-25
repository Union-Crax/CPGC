//! Order-1, Order-2, Order-4 statistical byte predictors.
//! Each maintains a frequency table and predicts P(next_byte | context).

use std::collections::HashMap;

const ALPHA: f32 = 1.0; // Laplace smoothing

// ---------------------------------------------------------------------------
// Order-1: P(next | prev)
// ---------------------------------------------------------------------------

pub struct Order1Model {
    counts: Box<[[u32; 256]; 256]>,
    totals: [u32; 256],
}

impl Order1Model {
    pub fn new() -> Box<Self> {
        Box::new(Self {
            counts: Box::new([[0u32; 256]; 256]),
            totals: [0u32; 256],
        })
    }

    pub fn predict(&self, prev: u8) -> [f32; 256] {
        let row = &self.counts[prev as usize];
        let total = self.totals[prev as usize] as f32 + 256.0 * ALPHA;
        let mut out = [0f32; 256];
        for (i, &c) in row.iter().enumerate() {
            out[i] = (c as f32 + ALPHA) / total;
        }
        out
    }

    pub fn update(&mut self, prev: u8, next: u8) {
        self.counts[prev as usize][next as usize] += 1;
        self.totals[prev as usize] += 1;
    }
}

// ---------------------------------------------------------------------------
// Order-2: P(next | prev2, prev1)  — sparse hashmap
// ---------------------------------------------------------------------------

pub struct Order2Model {
    counts: HashMap<u16, Box<[u32; 256]>>,
}

impl Order2Model {
    pub fn new() -> Self {
        Self { counts: HashMap::new() }
    }

    pub fn predict(&self, prev2: u8, prev1: u8) -> [f32; 256] {
        let key = ((prev2 as u16) << 8) | (prev1 as u16);
        if let Some(row) = self.counts.get(&key) {
            let total: u32 = row.iter().sum();
            let total_f = total as f32 + 256.0 * ALPHA;
            let mut out = [0f32; 256];
            for (i, &c) in row.iter().enumerate() {
                out[i] = (c as f32 + ALPHA) / total_f;
            }
            out
        } else {
            [1.0 / 256.0; 256]
        }
    }

    pub fn update(&mut self, prev2: u8, prev1: u8, next: u8) {
        let key = ((prev2 as u16) << 8) | (prev1 as u16);
        let row = self.counts.entry(key).or_insert_with(|| Box::new([0u32; 256]));
        row[next as usize] += 1;
    }
}

// ---------------------------------------------------------------------------
// Order-4: P(next | prev4..prev1)  — sparse hashmap, capped size
// ---------------------------------------------------------------------------

const ORDER4_CAP: usize = 1 << 20; // 1M entries max (~256MB cap)

pub struct Order4Model {
    counts: HashMap<u32, Box<[u32; 256]>>,
}

impl Order4Model {
    pub fn new() -> Self {
        Self { counts: HashMap::new() }
    }

    pub fn predict(&self, ctx4: u32) -> [f32; 256] {
        if let Some(row) = self.counts.get(&ctx4) {
            let total: u32 = row.iter().sum();
            let total_f = total as f32 + 256.0 * ALPHA;
            let mut out = [0f32; 256];
            for (i, &c) in row.iter().enumerate() {
                out[i] = (c as f32 + ALPHA) / total_f;
            }
            out
        } else {
            [1.0 / 256.0; 256]
        }
    }

    pub fn update(&mut self, ctx4: u32, next: u8) {
        if self.counts.len() >= ORDER4_CAP {
            return; // eviction not yet implemented — just stop inserting
        }
        let row = self.counts.entry(ctx4).or_insert_with(|| Box::new([0u32; 256]));
        row[next as usize] += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn order1_learns() {
        let mut m = Order1Model::new();
        // Need many updates to overcome Laplace smoothing across all 256 symbols.
        // After N updates: P(B|A) = (N+1) / (N+256) → 1 as N → ∞.
        // For P > 0.9: N+1 > 0.9*(N+256) → N > 2286
        for _ in 0..3000 {
            m.update(b'A', b'B');
        }
        let p = m.predict(b'A');
        assert!(p[b'B' as usize] > 0.9, "P(B|A) = {}", p[b'B' as usize]);
    }
}
