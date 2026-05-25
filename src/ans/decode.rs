//! rANS decoder.
//!
//! Reads the final encoder state from the last 4 bytes of the stream (LE),
//! then reads payload bytes backwards to renormalize after each decode step.
//! This produces symbols in the same forward order as they were passed to the encoder.

use crate::ans::table::{build_table, AnsTable, TABLE_SIZE};

const M: u32 = TABLE_SIZE;

pub struct AnsDecoder {
    state:         u32,
    data:          Vec<u8>,
    /// Index of the next payload byte to read (reads backwards toward 0).
    payload_pos:   usize,
    /// True once all payload bytes have been consumed.
    payload_empty: bool,
    table:         Box<AnsTable>,
}

impl AnsDecoder {
    /// Construct from bytes produced by `AnsEncoder::finish()`.
    /// Returns `None` if the stream is too short (< 4 bytes).
    pub fn new(compressed: &[u8], prob: &[f32; 256]) -> Option<Self> {
        let n = compressed.len();
        if n < 4 { return None; }
        // State is the last 4 bytes, LE
        let state = u32::from_le_bytes(
            compressed[n - 4..n].try_into().unwrap()
        );
        // Payload occupies compressed[0..n-4]
        let payload_len = n - 4;
        let (payload_pos, payload_empty) = if payload_len == 0 {
            (0, true)
        } else {
            (payload_len - 1, false)
        };
        Some(Self {
            state,
            data: compressed.to_vec(),
            payload_pos,
            payload_empty,
            table: build_table(prob),
        })
    }

    /// Update the probability table for subsequent decodes.
    pub fn update_table(&mut self, prob: &[f32; 256]) {
        self.table = build_table(prob);
    }

    /// Decode one symbol, returning `None` only if the initial stream was empty.
    pub fn decode(&mut self) -> Option<u8> {
        let slot = self.state % M;
        let sym  = self.table.slot_sym[slot as usize];
        let f    = self.table.freq[sym as usize];
        let c    = self.table.cumul[sym as usize];

        // Reverse encode step: x = f * (y / M) + (y % M) - c
        self.state = f * (self.state / M) + slot - c;
        // state is now in [f, f*256); renormalize into [M, M*256)
        while self.state < M {
            self.state = (self.state << 8) | (self.read_payload_byte() as u32);
        }
        Some(sym)
    }

    fn read_payload_byte(&mut self) -> u8 {
        if self.payload_empty {
            return 0;
        }
        let b = self.data[self.payload_pos];
        if self.payload_pos == 0 {
            self.payload_empty = true;
        } else {
            self.payload_pos -= 1;
        }
        b
    }
}
