//! Carryless binary arithmetic coder (the lpaq/zpaq range coder).
//!
//! Codes one bit at a time given a 12-bit probability `p = P(bit == 1)`,
//! `1 <= p <= 4095`. The `[x1, x2]` interval is split at
//! `xmid = x1 + (range >> 12) * p`; the top byte is emitted whenever `x1` and
//! `x2` agree on it. Because renormalisation guarantees `range >= 2^24` after
//! every step, `range >> 12 >= 2^12`, so `xmid` is always strictly inside the
//! interval and the coder never stalls.

/// Arithmetic encoder. Emits bytes into an internal buffer.
pub struct Encoder {
    x1: u32,
    x2: u32,
    out: Vec<u8>,
}

impl Encoder {
    pub fn new() -> Self {
        Self {
            x1: 0,
            x2: 0xffff_ffff,
            out: Vec::new(),
        }
    }

    /// Encode one bit with probability `p` (12-bit, `P(bit==1)`).
    #[inline]
    pub fn encode(&mut self, bit: i32, p: i32) {
        debug_assert!((1..=4095).contains(&p));
        let range = self.x2 - self.x1;
        let xmid = self.x1 + (range >> 12) * (p as u32);
        if bit != 0 {
            self.x2 = xmid;
        } else {
            self.x1 = xmid + 1;
        }
        // Renormalise: flush bytes that x1 and x2 now agree on.
        while (self.x1 ^ self.x2) & 0xff00_0000 == 0 {
            self.out.push((self.x2 >> 24) as u8);
            self.x1 <<= 8;
            self.x2 = (self.x2 << 8) | 0xff;
        }
    }

    /// Flush the remaining state and return the encoded byte stream.
    pub fn finish(mut self) -> Vec<u8> {
        // Emit all four bytes of x1 so the decoder can reconstruct the final
        // interval unambiguously regardless of how many bits remain.
        for _ in 0..4 {
            self.out.push((self.x1 >> 24) as u8);
            self.x1 <<= 8;
        }
        self.out
    }
}

/// Arithmetic decoder. Mirrors [`Encoder`] exactly.
pub struct Decoder<'a> {
    x1: u32,
    x2: u32,
    x: u32,
    inp: &'a [u8],
    pos: usize,
}

impl<'a> Decoder<'a> {
    pub fn new(inp: &'a [u8]) -> Self {
        let mut d = Self {
            x1: 0,
            x2: 0xffff_ffff,
            x: 0,
            inp,
            pos: 0,
        };
        // Prime x with the first four bytes (zero-padded if short).
        for _ in 0..4 {
            d.x = (d.x << 8) | (d.next_byte() as u32);
        }
        d
    }

    #[inline]
    fn next_byte(&mut self) -> u8 {
        let b = self.inp.get(self.pos).copied().unwrap_or(0);
        self.pos += 1;
        b
    }

    /// Decode one bit given the same probability `p` the encoder used.
    #[inline]
    pub fn decode(&mut self, p: i32) -> i32 {
        debug_assert!((1..=4095).contains(&p));
        let range = self.x2 - self.x1;
        let xmid = self.x1 + (range >> 12) * (p as u32);
        let bit = if self.x <= xmid { 1 } else { 0 };
        if bit != 0 {
            self.x2 = xmid;
        } else {
            self.x1 = xmid + 1;
        }
        while (self.x1 ^ self.x2) & 0xff00_0000 == 0 {
            self.x1 <<= 8;
            self.x2 = (self.x2 << 8) | 0xff;
            self.x = (self.x << 8) | (self.next_byte() as u32);
        }
        bit
    }
}
