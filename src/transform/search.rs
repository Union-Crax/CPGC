//! Search for the cheapest single-op transform for a chunk.

use crate::transform::primitives::TransformOp;
use crate::analyzer::entropy::entropy_bits;

/// Cost of storing a transform descriptor (1 byte overhead per chunk).
const TRANSFORM_COST_BITS: f32 = 8.0;

/// Candidate transforms tried by `find_best_transform`.
/// A transform's 1-based position in this slice becomes the block tag in the codec header
/// (tag 0x00 = no transform, 0x01 = CANDIDATES[0], …, 0x08 = CANDIDATES[7]).
pub static CANDIDATES: &[TransformOp] = &[
    TransformOp::Delta { step: 0 },        // tag 0x01
    TransformOp::Xor { mask: 0xFF },       // tag 0x02
    TransformOp::Add { value: 128 },       // tag 0x03
    TransformOp::BitRotate { bits: 1 },    // tag 0x04
    TransformOp::BitRotate { bits: 4 },    // tag 0x05
    TransformOp::Shift { bytes: 1 },       // tag 0x06
    TransformOp::Interleave { stride: 2 }, // tag 0x07
    TransformOp::Interleave { stride: 4 }, // tag 0x08
];

/// Find the best single transform op that reduces entropy, or return None.
pub fn find_best_transform(chunk: &[u8]) -> Option<(TransformOp, Vec<u8>)> {
    let base_entropy = entropy_bits(chunk) * chunk.len() as f32;

    let mut best_gain = TRANSFORM_COST_BITS;
    let mut best: Option<(TransformOp, Vec<u8>)> = None;

    for &op in CANDIDATES {
        let mut transformed = chunk.to_vec();
        op.apply(&mut transformed);
        let new_entropy = entropy_bits(&transformed) * transformed.len() as f32;
        let gain = base_entropy - new_entropy;
        if gain > best_gain {
            best_gain = gain;
            best = Some((op, transformed));
        }
    }

    best
}
