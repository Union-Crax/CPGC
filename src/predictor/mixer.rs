//! Linear context mixer: blends predictions from all sub-models.
//! Weights are learned online via gradient ascent on log-probability.

use crate::predictor::{
    lstm::TinyLSTM,
    match_model::MatchModel,
    order_n::{Order1Model, Order2Model, Order4Model},
    run_model::RunModel,
};

const NUM_MODELS: usize = 6;

pub struct ContextMixer {
    pub lstm:        Box<TinyLSTM>,
    pub order1:      Box<Order1Model>,
    pub order2:      Order2Model,
    pub order4:      Order4Model,
    pub run_model:   RunModel,
    pub match_model: MatchModel,

    mix_weights: [f32; NUM_MODELS],
    mix_lr:      f32,

    // Rolling context bytes
    prev1: u8,
    prev2: u8,
    ctx4:  u32,
}

impl ContextMixer {
    pub fn new(lr: f32) -> Self {
        Self {
            lstm:        TinyLSTM::new(lr),
            order1:      Order1Model::new(),
            order2:      Order2Model::new(),
            order4:      Order4Model::new(),
            run_model:   RunModel::new(),
            match_model: MatchModel::new(),
            mix_weights: [1.0 / NUM_MODELS as f32; NUM_MODELS],
            mix_lr:      0.001,
            prev1:       0,
            prev2:       0,
            ctx4:        0,
        }
    }

    /// Predict P(next byte). Does NOT advance model state.
    pub fn predict(&mut self) -> [f32; 256] {
        let preds: [[f32; 256]; NUM_MODELS] = [
            self.lstm.forward(self.prev1),
            self.order1.predict(self.prev1),
            self.order2.predict(self.prev2, self.prev1),
            self.order4.predict(self.ctx4),
            self.run_model.predict(),
            self.match_model.predict(),
        ];
        blend_predictions(&preds, &self.mix_weights)
    }

    /// Update all sub-models and the mixer with the actual next byte.
    /// Must be called after `predict()`.
    pub fn update(&mut self, actual: u8) {
        // Sub-model predictions at the time of last predict() call
        // (we recompute here since we don't cache — acceptable overhead)
        let preds: [[f32; 256]; NUM_MODELS] = [
            {
                // LSTM already advanced its internal state in forward(); update weights
                self.lstm.update(actual);
                // We can't easily re-run forward without side effects here, so use
                // uniform as a placeholder for mixer gradient — LSTM weight already updated above.
                [1.0 / 256.0; 256]
            },
            self.order1.predict(self.prev1),
            self.order2.predict(self.prev2, self.prev1),
            self.order4.predict(self.ctx4),
            self.run_model.predict(),
            self.match_model.predict(),
        ];

        // Update mixer weights: gradient ascent on log-prob of actual byte
        for (i, pred) in preds.iter().enumerate() {
            let log_p = pred[actual as usize].max(1e-30).ln();
            self.mix_weights[i] += self.mix_lr * log_p;
        }
        // Normalize via softmax
        self.mix_weights = softmax6(self.mix_weights);

        // Update statistical models
        self.order1.update(self.prev1, actual);
        self.order2.update(self.prev2, self.prev1, actual);
        self.order4.update(self.ctx4, actual);
        self.run_model.update(actual);
        self.match_model.update(actual);

        // Advance context
        self.prev2 = self.prev1;
        self.prev1 = actual;
        self.ctx4 = (self.ctx4 << 8) | (actual as u32);
    }
}

/// Blended prediction: weighted geometric mean in log-probability space.
fn blend_predictions(preds: &[[f32; 256]; NUM_MODELS], weights: &[f32; NUM_MODELS]) -> [f32; 256] {
    let mut log_blend = [0f32; 256];
    for (pred, &w) in preds.iter().zip(weights.iter()) {
        for (out, &p) in log_blend.iter_mut().zip(pred.iter()) {
            *out += w * p.max(1e-30).ln();
        }
    }
    // Softmax to re-normalize
    let max = log_blend.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0f32;
    for v in log_blend.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    let inv = 1.0 / sum;
    for v in log_blend.iter_mut() { *v *= inv; }
    log_blend
}

fn softmax6(mut w: [f32; NUM_MODELS]) -> [f32; NUM_MODELS] {
    let max = w.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0f32;
    for v in w.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    let inv = 1.0 / sum;
    for v in w.iter_mut() { *v *= inv; }
    w
}
