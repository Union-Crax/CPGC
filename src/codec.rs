//! Top-level compress / decompress orchestration.
//!
//! ## File Format (VERSION 2)
//!
//! ```text
//! [0..4]                      magic: "CPGC"
//! [4]                         version: 2
//! [5]                         flags: bit0 = has_passthrough, bit1 = has_transforms
//! [6..14]                     orig_len: u64 LE
//! [14..18]                    n_blocks: u32 LE  (= ceil(orig_len / BLOCK_SIZE))
//! [18..18+n_blocks]           block_tags: one byte per block
//!                               0x00 = LSTM+ANS, no transform
//!                               0x01–0x08 = LSTM+ANS on transformed data (1-indexed into CANDIDATES)
//!                               0xFF = passthrough (raw bytes, not run through LSTM)
//! [18+n_blocks..22+n_blocks]  passthrough_len: u32 LE
//! [22+n_blocks .. +passthrough_len]  raw bytes for passthrough blocks
//! [rest]                      rANS payload for all non-passthrough blocks (in block order)
//! ```
//!
//! Level < 5: no classify / transform pass; all block_tags = 0x00; passthrough_len = 0.
//! Level ≥ 5: content analyzer + transform search enabled.

use anyhow::{anyhow, Result};

use crate::analyzer::classifier::{classify, WINDOW_SIZE};
use crate::cm;
use crate::transform::search::{find_best_transform, CANDIDATES};

const MAGIC: &[u8; 4] = b"CPGC";
const VERSION: u8 = 4;
const TAG_NORMAL: u8 = 0x00;
const TAG_PASSTHROUGH: u8 = 0xFF;

// ---------------------------------------------------------------------------
// Compress
// ---------------------------------------------------------------------------

/// Compress `input` using the ContextMixer + rANS codec.
pub fn compress(input: &[u8], level: u8) -> Result<Vec<u8>> {
    compress_with_progress(input, level, |_, _| {})
}

/// Same as `compress` but calls `on_progress(bytes_encoded, total_bytes)` every 64 KB.
pub fn compress_with_progress(
    input: &[u8],
    level: u8,
    on_progress: impl Fn(usize, usize),
) -> Result<Vec<u8>> {
    let n = input.len();
    let n_blocks = if n == 0 { 0usize } else { (n + WINDOW_SIZE - 1) / WINDOW_SIZE };

    // ------------------------------------------------------------------
    // Step 1: Classify blocks; pass through incompressible blocks (every level)
    // and search transforms on structured ones (level ≥ 5).
    // ------------------------------------------------------------------
    let mut block_tags: Vec<u8> = vec![TAG_NORMAL; n_blocks];
    // Transformed data for transform-tagged blocks; None = use original chunk.
    let mut block_transformed: Vec<Option<Vec<u8>>> = vec![None; n_blocks];

    if n > 0 {
        let regions = classify(input);
        for (block_idx, region) in regions.iter().enumerate() {
            if block_idx >= n_blocks { break; }
            let start = block_idx * WINDOW_SIZE;
            let end = (start + WINDOW_SIZE).min(n);
            let chunk = &input[start..end];

            if region.passthrough {
                // Passing incompressible data through guards against expansion
                // at every level, not just ≥ 5.
                block_tags[block_idx] = TAG_PASSTHROUGH;
            } else if level >= 5 && region.use_transform {
                if let Some((op, transformed)) = find_best_transform(chunk) {
                    if let Some(tag) = op_to_tag(op) {
                        block_tags[block_idx] = tag;
                        block_transformed[block_idx] = Some(transformed);
                    }
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // Step 2: Build the two data streams
    // ------------------------------------------------------------------
    let mut to_encode: Vec<u8> = Vec::with_capacity(n);
    let mut passthrough_data: Vec<u8> = Vec::new();

    for block_idx in 0..n_blocks {
        let start = block_idx * WINDOW_SIZE;
        let end = (start + WINDOW_SIZE).min(n);
        let chunk = &input[start..end];
        let tag = block_tags[block_idx];

        if tag == TAG_PASSTHROUGH {
            passthrough_data.extend_from_slice(chunk);
        } else if let Some(ref tx) = block_transformed[block_idx] {
            to_encode.extend_from_slice(tx);
        } else {
            to_encode.extend_from_slice(chunk);
        }
    }

    // ------------------------------------------------------------------
    // Step 3: CPGC-NX context-mixing encode of the non-passthrough stream
    // ------------------------------------------------------------------
    // Progress is reported against the *full* input size so passthrough-heavy
    // files (e.g. already-compressed executables) show real MB/total MB.
    let grand_total = n.max(1);
    let pt_done = passthrough_data.len(); // passthrough bytes are instantly "done"
    if pt_done > 0 {
        on_progress(pt_done.min(grand_total), grand_total);
    }
    let ans_payload = cm::encode(&to_encode, level);
    on_progress(grand_total, grand_total);

    // ------------------------------------------------------------------
    // Step 4: Assemble output
    // ------------------------------------------------------------------
    let flags: u8 =
        (if passthrough_data.is_empty() { 0u8 } else { 1u8 })
        | (if block_tags.iter().any(|&t| t != TAG_NORMAL && t != TAG_PASSTHROUGH) { 2u8 } else { 0u8 });

    let total = 4 + 1 + 1 + 8 + 4 + n_blocks + 4 + passthrough_data.len() + ans_payload.len();
    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(MAGIC);
    out.push(VERSION);
    out.push(flags);
    out.extend_from_slice(&(n as u64).to_le_bytes());
    out.extend_from_slice(&(n_blocks as u32).to_le_bytes());
    out.extend_from_slice(&block_tags);
    out.extend_from_slice(&(passthrough_data.len() as u32).to_le_bytes());
    out.extend_from_slice(&passthrough_data);
    out.extend_from_slice(&ans_payload);
    Ok(out)
}

// ---------------------------------------------------------------------------
// Decompress
// ---------------------------------------------------------------------------

/// Decompress bytes produced by `compress()`.
pub fn decompress(input: &[u8]) -> Result<Vec<u8>> {
    // Minimum header: magic(4) + version(1) + flags(1) + orig_len(8) + n_blocks(4) + passthrough_len(4) = 22
    if input.len() < 22 {
        return Err(anyhow!("input too short for CPGC header"));
    }
    if &input[0..4] != MAGIC {
        return Err(anyhow!("invalid magic bytes"));
    }
    if input[4] != VERSION {
        return Err(anyhow!("unsupported version {} (expected {})", input[4], VERSION));
    }
    // input[5] = flags (reserved for decoder use; currently informational)
    let orig_len  = u64::from_le_bytes(input[6..14].try_into().unwrap()) as usize;
    let n_blocks  = u32::from_le_bytes(input[14..18].try_into().unwrap()) as usize;

    let tags_end = 18 + n_blocks;
    if input.len() < tags_end + 4 {
        return Err(anyhow!("truncated block table"));
    }
    let block_tags = &input[18..tags_end];
    let pt_len = u32::from_le_bytes(input[tags_end..tags_end + 4].try_into().unwrap()) as usize;

    let pt_start  = tags_end + 4;
    let ans_start = pt_start + pt_len;
    if input.len() < ans_start {
        return Err(anyhow!("truncated passthrough data"));
    }
    let passthrough_data = &input[pt_start..ans_start];
    let ans_payload      = &input[ans_start..];

    if orig_len == 0 {
        return Ok(Vec::new());
    }

    // ------------------------------------------------------------------
    // Decode the ANS stream for all non-passthrough blocks
    // ------------------------------------------------------------------
    let ans_byte_count: usize = block_tags.iter().enumerate().map(|(i, &tag)| {
        if tag == TAG_PASSTHROUGH { 0 }
        else {
            let start = i * WINDOW_SIZE;
            (start + WINDOW_SIZE).min(orig_len) - start
        }
    }).sum();

    let ans_decoded: Vec<u8> = if ans_byte_count > 0 {
        cm::decode(ans_payload, ans_byte_count)
    } else {
        Vec::new()
    };

    // ------------------------------------------------------------------
    // Reconstruct original byte order from block tags
    // ------------------------------------------------------------------
    let mut output = Vec::with_capacity(orig_len);
    let mut ans_pos = 0usize;
    let mut pt_pos  = 0usize;

    for (block_idx, &tag) in block_tags.iter().enumerate() {
        let start     = block_idx * WINDOW_SIZE;
        let block_len = (start + WINDOW_SIZE).min(orig_len) - start;

        if tag == TAG_PASSTHROUGH {
            if pt_pos + block_len > passthrough_data.len() {
                return Err(anyhow!("passthrough data underrun at block {}", block_idx));
            }
            output.extend_from_slice(&passthrough_data[pt_pos..pt_pos + block_len]);
            pt_pos += block_len;
        } else if tag == TAG_NORMAL {
            if ans_pos + block_len > ans_decoded.len() {
                return Err(anyhow!("ANS decoded data underrun at block {}", block_idx));
            }
            output.extend_from_slice(&ans_decoded[ans_pos..ans_pos + block_len]);
            ans_pos += block_len;
        } else {
            // Transform block: pull decoded bytes (still in transform space) then invert.
            if ans_pos + block_len > ans_decoded.len() {
                return Err(anyhow!("ANS decoded data underrun at transform block {}", block_idx));
            }
            let tag_idx = (tag - 1) as usize;
            if tag_idx >= CANDIDATES.len() {
                return Err(anyhow!("unknown transform tag 0x{:02x}", tag));
            }
            let mut block = ans_decoded[ans_pos..ans_pos + block_len].to_vec();
            ans_pos += block_len;
            CANDIDATES[tag_idx].invert(&mut block);
            output.extend_from_slice(&block);
        }
    }

    Ok(output)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Map a TransformOp to its 1-based CANDIDATES index (block tag value), or None.
fn op_to_tag(op: crate::transform::primitives::TransformOp) -> Option<u8> {
    CANDIDATES.iter().position(|&c| c == op).map(|i| (i + 1) as u8)
}


