//! Top-level compress / decompress orchestration.
//!
//! ## File Format (VERSION 8)
//!
//! ```text
//! [0..4]                      magic: "CPGC"
//! [4]                         version: 8
//! [5]                         flags: bit0 = has_passthrough, bit1 = has_transforms
//! [6..14]                     orig_len: u64 LE
//! [14..18]                    crc32: u32 LE  (CRC-32 of the original bytes)
//! [18..22]                    n_blocks: u32 LE  (= ceil(orig_len / WINDOW_SIZE))
//! [22..22+n_blocks]           block_tags: one byte per block
//!                               0x00 = context-mixed, no transform
//!                               0x01–0x08 = context-mixed on transformed data (1-indexed into CANDIDATES)
//!                               0xFF = passthrough (raw bytes, not run through the mixer)
//! [22+n_blocks..26+n_blocks]  passthrough_len: u32 LE
//! [26+n_blocks .. +passthrough_len]  raw bytes for passthrough blocks
//! [rest]                      context-mixer payload for all non-passthrough blocks (in block order)
//! ```
//!
//! The CRC-32 is verified after decoding, so a corrupt archive — or one written
//! by an incompatible model version — fails loudly instead of returning wrong
//! bytes. Level < 5: no classify / transform pass; all block_tags = 0x00;
//! passthrough_len = 0. Level ≥ 5: content analyzer + transform search enabled.

use anyhow::{anyhow, Result};

use crate::analyzer::classifier::{classify, WINDOW_SIZE};
use crate::cm;
use crate::checksum::crc32;
use crate::transform::search::{find_best_transform, CANDIDATES};

const MAGIC: &[u8; 4] = b"CPGC";
const VERSION: u8 = 8;
/// Smallest possible header: magic+ver+flags+orig_len+crc32+n_blocks+passthrough_len.
const HEADER_MIN: usize = 4 + 1 + 1 + 8 + 4 + 4 + 4;
const TAG_NORMAL: u8 = 0x00;
const TAG_PASSTHROUGH: u8 = 0xFF;

// ---------------------------------------------------------------------------
// Compress
// ---------------------------------------------------------------------------

/// Compress `input` using the CPGC-NX context-mixing codec.
pub fn compress(input: &[u8], level: u8) -> Result<Vec<u8>> {
    compress_with_control(input, level, &cm::Control::new())
}

/// Compress, polling `on_progress(bytes_done, total_bytes)` while a worker
/// thread does the actual work. Progress is real (driven by the byte counter
/// inside [`cm::Control`]), not an estimate.
pub fn compress_with_progress(
    input: &[u8],
    level: u8,
    on_progress: impl Fn(usize, usize),
) -> Result<Vec<u8>> {
    let total = input.len().max(1);
    let ctrl = cm::Control::new();
    let mut result: Option<Result<Vec<u8>>> = None;
    std::thread::scope(|s| {
        let handle = s.spawn(|| compress_with_control(input, level, &ctrl));
        while !handle.is_finished() {
            on_progress((ctrl.bytes_done() as usize).min(total), total);
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
        result = Some(handle.join().unwrap_or_else(|_| Err(anyhow!("worker panicked"))));
    });
    on_progress(total, total);
    result.unwrap()
}

/// Compress with a shared [`cm::Control`] for pause/resume/cancel and a live
/// byte counter (used by the GUI). Returns an error if the job is cancelled.
pub fn compress_with_control(input: &[u8], level: u8, ctrl: &cm::Control) -> Result<Vec<u8>> {
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
    let ans_payload = cm::encode_with_control(&to_encode, level, ctrl)
        .ok_or_else(|| anyhow!("compression cancelled"))?;

    // ------------------------------------------------------------------
    // Step 4: Assemble output
    // ------------------------------------------------------------------
    let flags: u8 =
        (if passthrough_data.is_empty() { 0u8 } else { 1u8 })
        | (if block_tags.iter().any(|&t| t != TAG_NORMAL && t != TAG_PASSTHROUGH) { 2u8 } else { 0u8 });

    let total = HEADER_MIN + n_blocks + passthrough_data.len() + ans_payload.len();
    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(MAGIC);
    out.push(VERSION);
    out.push(flags);
    out.extend_from_slice(&(n as u64).to_le_bytes());
    out.extend_from_slice(&crc32(input).to_le_bytes());
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
    decompress_with_control(input, &cm::Control::new())
}

/// Decompress with a shared [`cm::Control`] for pause/resume/cancel and a live
/// byte counter (used by the GUI). Returns an error if cancelled.
pub fn decompress_with_control(input: &[u8], ctrl: &cm::Control) -> Result<Vec<u8>> {
    if input.len() < HEADER_MIN {
        return Err(anyhow!("input too short for CPGC header"));
    }
    if &input[0..4] != MAGIC {
        return Err(anyhow!("invalid magic bytes"));
    }
    if input[4] != VERSION {
        return Err(anyhow!(
            "unsupported CPGC version {} (this build reads version {}). \
             Archives written by an older/newer build are not compatible.",
            input[4], VERSION
        ));
    }
    // input[5] = flags (reserved for decoder use; currently informational)
    let orig_len  = u64::from_le_bytes(input[6..14].try_into().unwrap()) as usize;
    let crc_expected = u32::from_le_bytes(input[14..18].try_into().unwrap());
    let n_blocks  = u32::from_le_bytes(input[18..22].try_into().unwrap()) as usize;

    let tags_end = 22 + n_blocks;
    if input.len() < tags_end + 4 {
        return Err(anyhow!("truncated block table"));
    }
    let block_tags = &input[22..tags_end];
    let pt_len = u32::from_le_bytes(input[tags_end..tags_end + 4].try_into().unwrap()) as usize;

    let pt_start  = tags_end + 4;
    let ans_start = pt_start + pt_len;
    if input.len() < ans_start {
        return Err(anyhow!("truncated passthrough data"));
    }
    let passthrough_data = &input[pt_start..ans_start];
    let ans_payload      = &input[ans_start..];

    if orig_len == 0 {
        if crc_expected != crc32(&[]) {
            return Err(anyhow!("checksum mismatch on empty archive (corrupt header)"));
        }
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
        cm::decode_with_control(ans_payload, ans_byte_count, ctrl)
            .ok_or_else(|| anyhow!("decompression cancelled"))?
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

    // Integrity check: a mismatch means the archive is corrupt or was written
    // by an incompatible model, so we must not hand back wrong bytes silently.
    let crc_actual = crc32(&output);
    if crc_actual != crc_expected {
        return Err(anyhow!(
            "checksum mismatch: archive is corrupt or was written by an \
             incompatible version (expected {:#010x}, got {:#010x})",
            crc_expected, crc_actual
        ));
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


