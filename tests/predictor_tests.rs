//! Tests to verify encoder and decoder produce identical model states.

use cpgc::predictor::lstm::TinyLSTM;

#[test]
fn encoder_decoder_sync() {
    let data: &[u8] = b"The quick brown fox jumps over the lazy dog. \
                         Pack my box with five dozen liquor jugs.";

    let mut enc_model = TinyLSTM::new(0.005);
    let mut dec_model = TinyLSTM::new(0.005);

    for &byte in data {
        let enc_probs = enc_model.forward(byte);
        let dec_probs = dec_model.forward(byte);

        // Encoder and decoder must see the same probability distribution
        for (a, b) in enc_probs.iter().zip(dec_probs.iter()) {
            assert!((a - b).abs() < 1e-6, "prob mismatch: {} vs {}", a, b);
        }

        enc_model.update(byte);
        dec_model.update(byte);

        // After update, hidden states must match
        #[cfg(debug_assertions)]
        {
            let (eh, ec) = enc_model.hidden_state();
            let (dh, dc) = dec_model.hidden_state();
            for (a, b) in eh.iter().zip(dh.iter()) {
                assert!((a - b).abs() < 1e-6, "hidden state h diverged");
            }
            for (a, b) in ec.iter().zip(dc.iter()) {
                assert!((a - b).abs() < 1e-6, "hidden state c diverged");
            }
        }
    }
}
