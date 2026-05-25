//! LZ match predictor.
//! Maintains a hash-based ring buffer and finds the longest recent match
//! for the current context, then predicts the byte following that match.

const HASH_BITS: usize = 16;
const HASH_SIZE: usize = 1 << HASH_BITS;
const WINDOW: usize = 1 << 20; // 1MB history window

pub struct MatchModel {
    history:    Vec<u8>,
    hash_table: [u32; HASH_SIZE], // maps hash → position in history
}

impl MatchModel {
    pub fn new() -> Self {
        Self {
            history:    Vec::with_capacity(WINDOW),
            hash_table: [u32::MAX; HASH_SIZE],
        }
    }

    /// Predict P(next) based on the best LZ match.
    /// Returns uniform distribution if no match is found.
    pub fn predict(&self) -> [f32; 256] {
        if self.history.len() < 4 {
            return [1.0 / 256.0; 256];
        }
        let ctx_len = self.history.len();
        let hash = self.context_hash(ctx_len);
        let match_pos = self.hash_table[hash & (HASH_SIZE - 1)];
        if match_pos == u32::MAX {
            return [1.0 / 256.0; 256];
        }
        let mp = match_pos as usize;
        if mp + 1 >= ctx_len {
            return [1.0 / 256.0; 256];
        }
        // The predicted byte is history[mp + 1] (with high confidence)
        let predicted_byte = self.history[mp + 1] as usize;
        let confidence = 0.9f32;
        let mut out = [(1.0 - confidence) / 255.0; 256];
        out[predicted_byte] = confidence + (1.0 - confidence) / 255.0;
        out
    }

    pub fn update(&mut self, byte: u8) {
        self.history.push(byte);
        if self.history.len() >= 4 {
            let len = self.history.len();
            let hash = self.context_hash(len);
            let slot = hash & (HASH_SIZE - 1);
            self.hash_table[slot] = (len - 1) as u32;
        }
        // Trim history to window size
        if self.history.len() > WINDOW {
            let drain = self.history.len() - WINDOW;
            self.history.drain(..drain);
        }
    }

    fn context_hash(&self, pos: usize) -> usize {
        // Hash last 4 bytes
        let n = self.history.len();
        if pos < 4 { return 0; }
        let b0 = self.history[n - 4] as usize;
        let b1 = self.history[n - 3] as usize;
        let b2 = self.history[n - 2] as usize;
        let b3 = self.history[n - 1] as usize;
        (b0 ^ (b1 << 3) ^ (b2 << 7) ^ (b3 << 13)).wrapping_mul(2654435761)
    }
}
