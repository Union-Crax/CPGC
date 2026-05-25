//! Bit-level I/O for the ANS bitstream.

pub struct BitWriter {
    buf: Vec<u8>,
    pending: u64,
    bits_in_pending: u8,
}

impl BitWriter {
    pub fn new() -> Self {
        Self { buf: Vec::new(), pending: 0, bits_in_pending: 0 }
    }

    /// Write `n` bits (n ≤ 56) from the low bits of `value`.
    pub fn write_bits(&mut self, value: u64, n: u8) {
        debug_assert!(n <= 56);
        self.pending |= (value & ((1u64 << n) - 1)) << self.bits_in_pending;
        self.bits_in_pending += n;
        while self.bits_in_pending >= 8 {
            self.buf.push(self.pending as u8);
            self.pending >>= 8;
            self.bits_in_pending -= 8;
        }
    }

    pub fn finish(mut self) -> Vec<u8> {
        if self.bits_in_pending > 0 {
            self.buf.push(self.pending as u8);
        }
        self.buf
    }
}

pub struct BitReader<'a> {
    data: &'a [u8],
    pos_byte: usize,
    pending: u64,
    bits_in_pending: u8,
}

impl<'a> BitReader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, pos_byte: 0, pending: 0, bits_in_pending: 0 }
    }

    /// Read `n` bits (n ≤ 56).
    pub fn read_bits(&mut self, n: u8) -> Option<u64> {
        debug_assert!(n <= 56);
        while self.bits_in_pending < n {
            if self.pos_byte >= self.data.len() {
                return None;
            }
            self.pending |= (self.data[self.pos_byte] as u64) << self.bits_in_pending;
            self.bits_in_pending += 8;
            self.pos_byte += 1;
        }
        let out = self.pending & ((1u64 << n) - 1);
        self.pending >>= n;
        self.bits_in_pending -= n;
        Some(out)
    }
}
