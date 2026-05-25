//! Run-length predictor.
//! Tracks the current run (repeated byte) and boosts its probability.

pub struct RunModel {
    last_byte: u8,
    run_len:   u32,
}

impl RunModel {
    pub fn new() -> Self {
        Self { last_byte: 0, run_len: 0 }
    }

    /// Predict P(next byte) given history.
    /// If we're in a run of `last_byte`, heavily favor continuation.
    pub fn predict(&self) -> [f32; 256] {
        let mut out = [1.0f32 / 256.0; 256];
        if self.run_len > 0 {
            // Confidence grows with run length, saturates quickly
            let weight = 1.0 - (-0.1 * self.run_len as f32).exp();
            let boost = weight * 0.99;
            let remaining = (1.0 - boost) / 255.0;
            for v in out.iter_mut() { *v = remaining; }
            out[self.last_byte as usize] = boost + remaining;
        }
        out
    }

    pub fn update(&mut self, byte: u8) {
        if byte == self.last_byte {
            self.run_len += 1;
        } else {
            self.last_byte = byte;
            self.run_len = 1;
        }
    }
}
