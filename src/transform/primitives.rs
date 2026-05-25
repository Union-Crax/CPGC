//! 11 reversible transform primitives.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum TransformOp {
    Shift { bytes: i16 },
    Xor { mask: u8 },
    Delta { step: i8 },
    Mirror,
    BitRotate { bits: u8 },
    Add { value: u8 },
    Subtract { value: u8 },
    BitMask { mask: u8 },
    Interleave { stride: u8 },
    Repeat { factor: u8 },
    Scale { numerator: u8, denominator: u8 },
}

impl TransformOp {
    /// Apply transform to a chunk of bytes (in-place).
    pub fn apply(&self, data: &mut Vec<u8>) {
        match *self {
            TransformOp::Shift { bytes } => {
                for v in data.iter_mut() {
                    *v = v.wrapping_add_signed(bytes as i8);
                }
            }
            TransformOp::Xor { mask } => {
                for v in data.iter_mut() { *v ^= mask; }
            }
            TransformOp::Delta { step } => {
                let mut prev = 0u8;
                for v in data.iter_mut() {
                    let next = v.wrapping_sub(prev);
                    prev = *v;
                    *v = next.wrapping_add_signed(step);
                }
            }
            TransformOp::Mirror => {
                data.reverse();
            }
            TransformOp::BitRotate { bits } => {
                let bits = bits % 8;
                for v in data.iter_mut() {
                    *v = v.rotate_left(bits as u32);
                }
            }
            TransformOp::Add { value } => {
                for v in data.iter_mut() { *v = v.wrapping_add(value); }
            }
            TransformOp::Subtract { value } => {
                for v in data.iter_mut() { *v = v.wrapping_sub(value); }
            }
            TransformOp::BitMask { mask } => {
                for v in data.iter_mut() { *v &= mask; }
            }
            TransformOp::Interleave { stride } => {
                if stride == 0 { return; }
                let num_ch = stride as usize;
                let len = data.len();
                let items = len / num_ch;
                let mut out = vec![0u8; len];
                for ch in 0..num_ch {
                    let mut dst = ch * items;
                    let mut src = ch;
                    while src < len {
                        out[dst] = data[src];
                        src += num_ch;
                        dst += 1;
                    }
                }
                *data = out;
            }
            TransformOp::Repeat { factor } => {
                if factor <= 1 { return; }
                let orig = data.clone();
                data.clear();
                for _ in 0..factor { data.extend_from_slice(&orig); }
            }
            TransformOp::Scale { numerator, denominator } => {
                if denominator == 0 { return; }
                for v in data.iter_mut() {
                    *v = ((*v as u32 * numerator as u32) / denominator as u32).min(255) as u8;
                }
            }
        }
    }

    /// Invert a previously applied transform (in-place).
    pub fn invert(&self, data: &mut Vec<u8>) {
        match *self {
            TransformOp::Shift { bytes } => {
                TransformOp::Shift { bytes: -bytes }.apply(data);
            }
            TransformOp::Xor { mask: _ } => { self.apply(data); } // XOR is its own inverse
            TransformOp::Delta { step } => {
                let mut prev = 0u8;
                for v in data.iter_mut() {
                    let shifted = v.wrapping_sub_signed(step);
                    let orig = shifted.wrapping_add(prev);
                    prev = orig;
                    *v = orig;
                }
            }
            TransformOp::Mirror => { self.apply(data); }
            TransformOp::BitRotate { bits } => {
                let bits = bits % 8;
                for v in data.iter_mut() {
                    *v = v.rotate_right(bits as u32);
                }
            }
            TransformOp::Add { value } => {
                TransformOp::Subtract { value }.apply(data);
            }
            TransformOp::Subtract { value } => {
                TransformOp::Add { value }.apply(data);
            }
            TransformOp::BitMask { .. } => {
                // Not generally invertible — no-op inversion
            }
            TransformOp::Interleave { stride } => {
                if stride == 0 { return; }
                let num_ch = stride as usize;
                let len = data.len();
                let items = len / num_ch;
                let mut out = vec![0u8; len];
                for ch in 0..num_ch {
                    let mut src = ch * items;
                    let mut dst = ch;
                    while dst < len {
                        out[dst] = data[src];
                        src += 1;
                        dst += num_ch;
                    }
                }
                *data = out;
            }
            TransformOp::Repeat { .. } => {
                // Inversion not generally possible without knowing original length — no-op
            }
            TransformOp::Scale { numerator, denominator } => {
                if numerator == 0 { return; }
                for v in data.iter_mut() {
                    *v = ((*v as u32 * denominator as u32) / numerator as u32).min(255) as u8;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(op: TransformOp, data: &[u8]) {
        let mut d = data.to_vec();
        op.apply(&mut d);
        op.invert(&mut d);
        assert_eq!(d, data, "{:?} roundtrip failed", op);
    }

    #[test]
    fn xor_roundtrip() { roundtrip(TransformOp::Xor { mask: 0xA5 }, b"hello world"); }
    #[test]
    fn add_roundtrip() { roundtrip(TransformOp::Add { value: 42 }, b"hello world"); }
    #[test]
    fn delta_roundtrip() { roundtrip(TransformOp::Delta { step: 0 }, b"hello world"); }
    #[test]
    fn mirror_roundtrip() { roundtrip(TransformOp::Mirror, b"hello world"); }
    #[test]
    fn bitrotate_roundtrip() { roundtrip(TransformOp::BitRotate { bits: 3 }, b"hello world"); }
    #[test]
    fn shift_roundtrip() { roundtrip(TransformOp::Shift { bytes: 5 }, b"hello world"); }
    #[test]
    fn interleave_roundtrip() { roundtrip(TransformOp::Interleave { stride: 4 }, b"ABCDABCDABCDABCD"); }
}
