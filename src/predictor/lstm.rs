//! TinyLSTM: 64-hidden-unit online LSTM predictor.
//!
//! Input:  last byte (as embedding index)
//! Output: probability distribution over next byte [256 floats, sum = 1]
//!
//! The model updates its weights after every predicted byte (online SGD with
//! momentum), so encoder and decoder stay in sync without transmitting weights.

/// Byte embedding dimension and hidden state size.
const EMBED: usize = 64;
const HIDDEN: usize = 64;

/// Combined gate width: 4 gates × HIDDEN = 256
const GATES: usize = 4 * HIDDEN; // 256

/// Input to gate matrix input width: EMBED + HIDDEN
const XH: usize = EMBED + HIDDEN; // 128

pub struct TinyLSTM {
    // Weights (f32 for now; see plan §Phase 5 for int8 quantization)
    embed: [[f32; EMBED]; 256],           // 256 × 64
    w_gates: [[f32; XH]; GATES],          // 256 × 128
    b_gates: [f32; GATES],                // 256
    w_out: [[f32; HIDDEN]; 256],          // 256 × 64
    b_out: [f32; 256],

    // Runtime state
    h: [f32; HIDDEN],
    c: [f32; HIDDEN],
    last_byte: u8,

    // SGD with momentum
    lr: f32,
    mom_gates: Box<[[f32; XH]; GATES]>,
    mom_out:   Box<[[f32; HIDDEN]; 256]>,
    mom_b_gates: [f32; GATES],
    mom_b_out:   [f32; 256],
    mom_embed:   Box<[[f32; EMBED]; 256]>,

    // Saved activations for BPTT-1 backward pass
    last_x: [f32; EMBED],
    last_xh: [f32; XH],
    last_i: [f32; HIDDEN],
    last_f: [f32; HIDDEN],
    last_g: [f32; HIDDEN],
    last_o: [f32; HIDDEN],
    last_c_prev: [f32; HIDDEN],
    last_h_prev: [f32; HIDDEN],
    last_logits: [f32; 256],
}

impl TinyLSTM {
    pub fn new(lr: f32) -> Box<Self> {
        // Xavier-ish init: small random values
        // Using a simple deterministic LCG so init is reproducible.
        let mut lcg: u64 = 0x_dead_beef_cafe_1234;
        let mut rng = move || -> f32 {
            lcg = lcg.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let bits = ((lcg >> 33) as u32) as f32 / (u32::MAX as f32) - 0.5;
            bits * 0.1
        };

        let mut lstm = Box::new(Self {
            embed:       [[0f32; EMBED]; 256],
            w_gates:     [[0f32; XH]; GATES],
            b_gates:     [0f32; GATES],
            w_out:       [[0f32; HIDDEN]; 256],
            b_out:       [0f32; 256],
            h:           [0f32; HIDDEN],
            c:           [0f32; HIDDEN],
            last_byte:   0,
            lr,
            mom_gates:   Box::new([[0f32; XH]; GATES]),
            mom_out:     Box::new([[0f32; HIDDEN]; 256]),
            mom_b_gates: [0f32; GATES],
            mom_b_out:   [0f32; 256],
            mom_embed:   Box::new([[0f32; EMBED]; 256]),
            last_x:      [0f32; EMBED],
            last_xh:     [0f32; XH],
            last_i:      [0f32; HIDDEN],
            last_f:      [0f32; HIDDEN],
            last_g:      [0f32; HIDDEN],
            last_o:      [0f32; HIDDEN],
            last_c_prev: [0f32; HIDDEN],
            last_h_prev: [0f32; HIDDEN],
            last_logits: [0f32; 256],
        });

        for row in lstm.embed.iter_mut() {
            for v in row.iter_mut() { *v = rng(); }
        }
        for row in lstm.w_gates.iter_mut() {
            for v in row.iter_mut() { *v = rng(); }
        }
        for row in lstm.w_out.iter_mut() {
            for v in row.iter_mut() { *v = rng(); }
        }
        lstm
    }

    /// Forward pass: given current byte, return P(next byte) [256 floats].
    /// Call `update()` afterwards with the actual next byte.
    pub fn forward(&mut self, byte: u8) -> [f32; 256] {
        // 1. Embed input byte
        let x = self.embed[byte as usize];

        // 2. Concatenate [x, h] → xh (128 floats)
        let mut xh = [0f32; XH];
        xh[..EMBED].copy_from_slice(&x);
        xh[EMBED..].copy_from_slice(&self.h);

        // 3. Gate pre-activations: gates = W_gates @ xh + b_gates  (SIMD)
        let gates_pre = matmul_256x128(&self.w_gates, &xh, &self.b_gates);

        // 4. Apply activations
        let mut i_gate = [0f32; HIDDEN];
        let mut f_gate = [0f32; HIDDEN];
        let mut g_gate = [0f32; HIDDEN];
        let mut o_gate = [0f32; HIDDEN];
        for k in 0..HIDDEN {
            i_gate[k] = sigmoid(gates_pre[k]);
            f_gate[k] = sigmoid(gates_pre[HIDDEN + k]);
            g_gate[k] = tanh_f(gates_pre[2 * HIDDEN + k]);
            o_gate[k] = sigmoid(gates_pre[3 * HIDDEN + k]);
        }

        // 5. Update cell & hidden
        let c_prev = self.c;
        let h_prev = self.h;
        let mut c_new = [0f32; HIDDEN];
        let mut h_new = [0f32; HIDDEN];
        for k in 0..HIDDEN {
            c_new[k] = f_gate[k] * c_prev[k] + i_gate[k] * g_gate[k];
            h_new[k] = o_gate[k] * tanh_f(c_new[k]);
        }
        self.c = c_new;
        self.h = h_new;

        // 6. Project to logits: w_out @ h + b_out  (SIMD)
        let logits = matmul_256x64(&self.w_out, &h_new, &self.b_out);
        let probs = softmax(logits);

        // Save activations for backward
        self.last_byte   = byte;
        self.last_x      = x;
        self.last_xh     = xh;
        self.last_i      = i_gate;
        self.last_f      = f_gate;
        self.last_g      = g_gate;
        self.last_o      = o_gate;
        self.last_c_prev = c_prev;
        self.last_h_prev = h_prev;
        self.last_logits = probs;

        probs
    }

    /// Truncated BPTT-1 update given the actual byte that followed.
    pub fn update(&mut self, actual: u8) {
        const MOMENTUM: f32 = 0.9;

        // --- Output layer gradient ---
        // grad_logits[i] = pred[i] - 1_{i == actual}
        let mut d_logits = self.last_logits;
        d_logits[actual as usize] -= 1.0;

        // grad_h from output projection: d_h = W_out^T @ d_logits
        let mut d_h = [0f32; HIDDEN];
        for (sym, row) in self.w_out.iter().enumerate() {
            let dl = d_logits[sym];
            for k in 0..HIDDEN {
                d_h[k] += row[k] * dl;
            }
        }

        // Update W_out and b_out
        for (sym, row) in self.w_out.iter_mut().enumerate() {
            let dl = d_logits[sym];
            for k in 0..HIDDEN {
                let g = dl * self.h[k];
                self.mom_out[sym][k] = MOMENTUM * self.mom_out[sym][k] + g;
                row[k] -= self.lr * self.mom_out[sym][k];
            }
            self.mom_b_out[sym] = MOMENTUM * self.mom_b_out[sym] + d_logits[sym];
            self.b_out[sym] -= self.lr * self.mom_b_out[sym];
        }

        // --- LSTM backward (1-step) ---
        // d_h flows into o_gate and tanh(c)
        let c = self.c; // current cell (post-update)
        let mut d_c = [0f32; HIDDEN];
        let mut d_gates = [0f32; GATES];

        for k in 0..HIDDEN {
            let tc = tanh_f(c[k]);
            let d_o = d_h[k] * tc;
            let d_tanh_c = d_h[k] * self.last_o[k];
            d_c[k] = d_tanh_c * (1.0 - tc * tc);

            let d_i = d_c[k] * self.last_g[k];
            let d_f = d_c[k] * self.last_c_prev[k];
            let d_g = d_c[k] * self.last_i[k];

            d_gates[k]              = d_i * self.last_i[k] * (1.0 - self.last_i[k]); // sigmoid'
            d_gates[HIDDEN + k]     = d_f * self.last_f[k] * (1.0 - self.last_f[k]);
            d_gates[2 * HIDDEN + k] = d_g * (1.0 - self.last_g[k] * self.last_g[k]); // tanh'
            d_gates[3 * HIDDEN + k] = d_o * self.last_o[k] * (1.0 - self.last_o[k]);
        }

        // Update W_gates and b_gates
        for (i, row) in self.w_gates.iter_mut().enumerate() {
            let dg = d_gates[i];
            for j in 0..XH {
                let g = dg * self.last_xh[j];
                self.mom_gates[i][j] = MOMENTUM * self.mom_gates[i][j] + g;
                row[j] -= self.lr * self.mom_gates[i][j];
            }
            self.mom_b_gates[i] = MOMENTUM * self.mom_b_gates[i] + dg;
            self.b_gates[i] -= self.lr * self.mom_b_gates[i];
        }

        // Update embedding for the last input byte.
        // d_x = W_gates[:, :EMBED]^T @ d_gates  (only EMBED columns needed)
        let b = self.last_byte as usize;
        let mut d_x = [0f32; EMBED];
        for i in 0..GATES {
            let dg = d_gates[i];
            for j in 0..EMBED {
                d_x[j] += self.w_gates[i][j] * dg;
            }
        }
        for k in 0..EMBED {
            self.mom_embed[b][k] = MOMENTUM * self.mom_embed[b][k] + d_x[k];
            self.embed[b][k] -= self.lr * self.mom_embed[b][k];
        }
    }

    /// Reset hidden/cell state (e.g. between files in an archive).
    pub fn reset_state(&mut self) {
        self.h = [0f32; HIDDEN];
        self.c = [0f32; HIDDEN];
    }

    /// Snapshot current weights as int8 for compact persistent storage.
    ///
    /// Uses per-row symmetric quantization: `q[j] = round(w[j] / scale)` where
    /// `scale = max(|w[j]|) / 127`.  Reconstruction error ≤ scale/2 per element.
    ///
    /// Memory breakdown (f32 → int8):
    ///   w_gates:  128KB → 32KB   w_out:   64KB → 16KB   embed:   64KB → 16KB
    ///   Total weights: ~256KB → ~64KB  (4× reduction)
    pub fn quantize_snapshot(&self) -> Box<QuantizedSnapshot> {
        let mut snap = Box::new(QuantizedSnapshot {
            w_gates:       Box::new([[0i8; XH]; GATES]),
            w_gates_scale: [0f32; GATES],
            w_out:         Box::new([[0i8; HIDDEN]; 256]),
            w_out_scale:   [0f32; 256],
            embed:         Box::new([[0i8; EMBED]; 256]),
            embed_scale:   [0f32; 256],
            b_gates: self.b_gates,
            b_out:   self.b_out,
            h:       self.h,
            c:       self.c,
        });
        for i in 0..GATES {
            let (q, s) = quantize_row_128(&self.w_gates[i]);
            snap.w_gates[i]       = q;
            snap.w_gates_scale[i] = s;
        }
        for i in 0..256 {
            let (q, s) = quantize_row_64(&self.w_out[i]);
            snap.w_out[i]       = q;
            snap.w_out_scale[i] = s;
        }
        for i in 0..256 {
            let (q, s) = quantize_row_64(&self.embed[i]);
            snap.embed[i]       = q;
            snap.embed_scale[i] = s;
        }
        snap
    }

    #[cfg(debug_assertions)]
    pub fn hidden_state(&self) -> ([f32; HIDDEN], [f32; HIDDEN]) {
        (self.h, self.c)
    }
}

// ---------------------------------------------------------------------------
// int8 quantization helpers (Phase 5 — compact model storage)
// ---------------------------------------------------------------------------

/// Compact snapshot of LSTM weights quantized to int8 with per-row scales.
/// Used for persistent storage; decoder reconstructs weights via dequantize.
pub struct QuantizedSnapshot {
    pub w_gates:       Box<[[i8; XH]; GATES]>,
    pub w_gates_scale: [f32; GATES],
    pub w_out:         Box<[[i8; HIDDEN]; 256]>,
    pub w_out_scale:   [f32; 256],
    pub embed:         Box<[[i8; EMBED]; 256]>,
    pub embed_scale:   [f32; 256],
    // Kept as f32 — these are small (< 4KB total)
    pub b_gates: [f32; GATES],
    pub b_out:   [f32; 256],
    pub h:       [f32; HIDDEN],
    pub c:       [f32; HIDDEN],
}

fn quantize_row_128(row: &[f32; XH]) -> ([i8; XH], f32) {
    let max_abs = row.iter().cloned().fold(0f32, |a, x| a.max(x.abs()));
    if max_abs == 0.0 { return ([0i8; XH], 1.0); }
    let scale = max_abs / 127.0;
    let inv   = 1.0 / scale;
    let mut q = [0i8; XH];
    for (i, &v) in row.iter().enumerate() {
        q[i] = (v * inv).clamp(-127.0, 127.0).round() as i8;
    }
    (q, scale)
}

fn quantize_row_64(row: &[f32; HIDDEN]) -> ([i8; HIDDEN], f32) {
    let max_abs = row.iter().cloned().fold(0f32, |a, x| a.max(x.abs()));
    if max_abs == 0.0 { return ([0i8; HIDDEN], 1.0); }
    let scale = max_abs / 127.0;
    let inv   = 1.0 / scale;
    let mut q = [0i8; HIDDEN];
    for (i, &v) in row.iter().enumerate() {
        q[i] = (v * inv).clamp(-127.0, 127.0).round() as i8;
    }
    (q, scale)
}

// ---------------------------------------------------------------------------
// SIMD matrix-vector multiply helpers (via `wide::f32x8`)
// Dimensions are compile-time constants (GATES=256, XH=128, HIDDEN=64),
// both multiples of 8 — no remainder handling needed.
// ---------------------------------------------------------------------------

/// out[i] = b[i] + dot(w[i], x)  for each of 256 rows with 128 columns.
#[inline(always)]
fn matmul_256x128(w: &[[f32; XH]; GATES], x: &[f32; XH], b: &[f32; GATES]) -> [f32; GATES] {
    use wide::f32x8;
    let mut out = *b;
    for (i, row) in w.iter().enumerate() {
        let mut acc = f32x8::splat(0.0);
        for j in (0..XH).step_by(8) {
            let wv = f32x8::from(<[f32; 8]>::try_from(&row[j..j + 8]).unwrap());
            let xv = f32x8::from(<[f32; 8]>::try_from(&x[j..j + 8]).unwrap());
            acc += wv * xv;
        }
        out[i] += acc.reduce_add();
    }
    out
}

/// out[i] = b[i] + dot(w[i], x)  for each of 256 rows with 64 columns.
#[inline(always)]
fn matmul_256x64(w: &[[f32; HIDDEN]; 256], x: &[f32; HIDDEN], b: &[f32; 256]) -> [f32; 256] {
    use wide::f32x8;
    let mut out = *b;
    for (i, row) in w.iter().enumerate() {
        let mut acc = f32x8::splat(0.0);
        for j in (0..HIDDEN).step_by(8) {
            let wv = f32x8::from(<[f32; 8]>::try_from(&row[j..j + 8]).unwrap());
            let xv = f32x8::from(<[f32; 8]>::try_from(&x[j..j + 8]).unwrap());
            acc += wv * xv;
        }
        out[i] += acc.reduce_add();
    }
    out
}

// ---------------------------------------------------------------------------
// Activation functions
// ---------------------------------------------------------------------------

#[inline(always)]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline(always)]
fn tanh_f(x: f32) -> f32 {
    x.tanh()
}

fn softmax(mut logits: [f32; 256]) -> [f32; 256] {
    // Numerically stable: subtract max before exp
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0f32;
    for v in logits.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    let inv = 1.0 / sum;
    for v in logits.iter_mut() { *v *= inv; }
    logits
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probs_sum_to_one() {
        let mut model = TinyLSTM::new(0.005);
        let probs = model.forward(b'A');
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "probs sum = {}", sum);
    }

    #[test]
    fn update_does_not_panic() {
        let mut model = TinyLSTM::new(0.005);
        let _probs = model.forward(b'A');
        model.update(b'B');
        let probs2 = model.forward(b'B');
        let sum: f32 = probs2.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn learns_repeated_byte() {
        // Feed "AAAAAAA..." — model should raise P('A' | context)
        let mut model = TinyLSTM::new(0.01);
        let runs = 500;
        let mut p_a_initial = 0f32;
        let mut p_a_final = 0f32;
        for i in 0..runs {
            let probs = model.forward(b'A');
            if i == 0 { p_a_initial = probs[b'A' as usize]; }
            if i == runs - 1 { p_a_final = probs[b'A' as usize]; }
            model.update(b'A');
        }
        assert!(p_a_final > p_a_initial, "model did not learn: {:.4} → {:.4}", p_a_initial, p_a_final);
    }
}
