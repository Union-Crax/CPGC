//! Adversarial input tests: random bytes, high-entropy data, all-same bytes.

#[test]
fn random_bytes_do_not_panic() {
    use cpgc::predictor::lstm::TinyLSTM;
    let mut model = TinyLSTM::new(0.005);
    // Pseudo-random via LCG
    let mut lcg: u32 = 0xDEADBEEF;
    for _ in 0..10_000 {
        lcg = lcg.wrapping_mul(1664525).wrapping_add(1013904223);
        let byte = (lcg >> 24) as u8;
        let _probs = model.forward(byte);
        model.update(byte);
    }
}

#[test]
fn all_zeros_do_not_panic() {
    use cpgc::predictor::lstm::TinyLSTM;
    let mut model = TinyLSTM::new(0.005);
    for _ in 0..1000 {
        let _probs = model.forward(0);
        model.update(0);
    }
}

#[test]
fn transform_roundtrip_adversarial() {
    use cpgc::transform::primitives::TransformOp;
    // All byte values
    let data: Vec<u8> = (0u8..=255).collect();
    for op in [
        TransformOp::Xor { mask: 0x55 },
        TransformOp::Add { value: 200 },
        TransformOp::Subtract { value: 200 },
        TransformOp::BitRotate { bits: 7 },
        TransformOp::Mirror,
        TransformOp::Delta { step: 0 },
    ] {
        let mut d = data.clone();
        op.apply(&mut d);
        op.invert(&mut d);
        assert_eq!(d, data, "adversarial roundtrip failed for {:?}", op);
    }
}
