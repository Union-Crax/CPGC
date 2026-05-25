//! LZ match predictor.
//!
//! Uses a power-of-two ring buffer so the history never needs to be drained
//! (the old Vec + drain approach had a position-drift bug after the first drain).
//!
//! Window : 8 MB  — catches long-range repeats in game assets / executables.
//! Hash   : 1 M entries (4 MB) — enough for 8 MB of distinct 4-byte contexts.

const HASH_BITS: usize = 20;
const HASH_SIZE: usize = 1 << HASH_BITS; // 1 M entries

const WINDOW_LOG: usize = 23;
const WINDOW: usize     = 1 << WINDOW_LOG; // 8 MB
const WMASK: usize      = WINDOW - 1;

pub struct MatchModel {
    buf:        Box<[u8]>,   // ring buffer, always WINDOW bytes
    hash_table: Box<[u32]>,  // slot → absolute position of matching 4-byte context
    pos:        usize,       // total bytes written (absolute, never resets)
}

impl MatchModel {
    pub fn new() -> Self {
        Self {
            buf:        vec![0u8; WINDOW].into_boxed_slice(),
            hash_table: vec![u32::MAX; HASH_SIZE].into_boxed_slice(),
            pos:        0,
        }
    }

    /// Hash the 4 bytes ending at absolute position `end` (inclusive).
    #[inline]
    fn hash4(&self, end: usize) -> usize {
        let b0 = self.buf[end.wrapping_sub(3) & WMASK] as usize;
        let b1 = self.buf[end.wrapping_sub(2) & WMASK] as usize;
        let b2 = self.buf[end.wrapping_sub(1) & WMASK] as usize;
        let b3 = self.buf[end               & WMASK] as usize;
        (b0 ^ (b1 << 3) ^ (b2 << 7) ^ (b3 << 13))
            .wrapping_mul(2654435761)
            & (HASH_SIZE - 1)
    }

    /// Predict P(next byte) from the best LZ match.
    pub fn predict(&self) -> [f32; 256] {
        if self.pos < 4 {
            return [1.0 / 256.0; 256];
        }
        let slot = self.hash4(self.pos - 1);
        let mp   = self.hash_table[slot];
        if mp == u32::MAX {
            return [1.0 / 256.0; 256];
        }
        let mp = mp as usize;
        // mp is the last byte of the stored context; predicted byte is at mp+1.
        // Guard: mp+1 must be in the past and within the 8 MB window.
        if mp + 1 >= self.pos || self.pos.saturating_sub(mp + 1) > WINDOW {
            return [1.0 / 256.0; 256];
        }
        let predicted = self.buf[(mp + 1) & WMASK] as usize;
        let conf = 0.9f32;
        let mut out = [(1.0 - conf) / 255.0; 256];
        out[predicted] = conf + (1.0 - conf) / 255.0;
        out
    }

    pub fn update(&mut self, byte: u8) {
        self.buf[self.pos & WMASK] = byte;
        if self.pos >= 3 {
            let slot = self.hash4(self.pos);
            // Truncate to u32; stale entries >4 GB back are safely rejected in predict().
            self.hash_table[slot] = self.pos as u32;
        }
        self.pos += 1;
    }
}
