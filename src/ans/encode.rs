//! rANS encoder — buffered 2-pass.
//!
//! rANS is naturally LIFO: encoding order is reversed relative to decoding order.
//! To decode in forward order (s0, s1, ...), we must encode in reverse (sN, ..., s0).
//!
//! The encoder buffers (sym, freq, cumul) tuples as `encode()` is called.
//! `finish()` runs the actual rANS in reverse, then appends the final state
//! (4 bytes LE) to the end of the output buffer.
//!
//! Output layout: [payload bytes...] [state: 4 bytes LE]
//! Decoder reads state from the last 4 bytes, then reads payload bytes backwards.

use crate::ans::table::{build_table, AnsTable, TABLE_SIZE};

const M: u32 = TABLE_SIZE;

pub struct AnsEncoder {
    /// Buffered encode items: (symbol, freq[sym], cumul[sym]) at encoding time.
    items:         Vec<(u8, u32, u32)>,
    current_table: Box<AnsTable>,
}

impl AnsEncoder {
    pub fn new(prob: &[f32; 256]) -> Self {
        Self { items: Vec::new(), current_table: build_table(prob) }
    }

    /// Update the current probability table.
    /// Captured f/c values for already-buffered symbols are unaffected.
    pub fn update_table(&mut self, prob: &[f32; 256]) {
        self.current_table = build_table(prob);
    }

    /// Buffer one symbol. Captures freq and cumul from the current table.
    pub fn encode(&mut self, symbol: u8) {
        let f = self.current_table.freq[symbol as usize];
        let c = self.current_table.cumul[symbol as usize];
        debug_assert!(f > 0, "symbol {} has zero frequency", symbol);
        self.items.push((symbol, f, c));
    }

    /// Encode all buffered symbols in reverse order (rANS LIFO property)
    /// and return the compressed bytes.
    ///
    /// Output: [payload_bytes...][state: 4 bytes LE]
    pub fn finish(self) -> Vec<u8> {
        let mut state: u32 = M;
        let mut output: Vec<u8> = Vec::new();

        // Encode buffered items in REVERSE (last encoded = first decoded)
        for &(_, f, c) in self.items.iter().rev() {
            // Flush bytes while state would overflow after encoding
            let max_for_sym = f.saturating_mul(256);
            while state >= max_for_sym {
                output.push((state & 0xFF) as u8);
                state >>= 8;
            }
            // x' = (x / f) * M + c + (x % f)  in [M, M*256)
            state = (state / f) * M + c + (state % f);
        }

        // Append final state (4 bytes, little-endian)
        output.extend_from_slice(&state.to_le_bytes());
        output
    }
}
