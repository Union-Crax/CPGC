//! Region classifier: produces a map of preprocessing flags per 4KB window.

use crate::analyzer::entropy::{entropy_bits, is_truly_incompressible};
use crate::analyzer::magic::is_incompressible;

pub const WINDOW_SIZE: usize = 4096;

#[derive(Clone, Debug, Default)]
pub struct Region {
    pub offset:      u32,
    pub passthrough: bool,  // skip compression entirely
    pub use_delta:   u8,    // 0 = no delta, 1-4 = delta order
    pub use_transform: bool,
}

/// Classify input data into regions with preprocessing hints.
pub fn classify(data: &[u8]) -> Vec<Region> {
    let mut regions = Vec::new();
    let mut offset = 0usize;

    // Check file-level magic
    let file_incompressible = is_incompressible(data);

    while offset < data.len() {
        let end = (offset + WINDOW_SIZE).min(data.len());
        let window = &data[offset..end];

        let mut region = Region { offset: offset as u32, ..Default::default() };

        if file_incompressible || is_truly_incompressible(window) {
            region.passthrough = true;
        } else {
            // Heuristic: if delta-encoding reduces entropy, flag it
            let e0 = entropy_bits(window);
            let delta: Vec<u8> = window.windows(2).map(|w| w[1].wrapping_sub(w[0])).collect();
            if !delta.is_empty() {
                let e1 = entropy_bits(&delta);
                if e1 < e0 - 0.5 {
                    region.use_delta = 1;
                }
            }
            // Flag structured binary regions for transform search (done in codec layer)
            if e0 < 6.0 && region.use_delta == 0 {
                region.use_transform = true;
            }
        }

        regions.push(region);
        offset += WINDOW_SIZE;
    }

    regions
}
