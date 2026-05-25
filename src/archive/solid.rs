//! Solid archive mode: compress all files into one continuous CPGC stream.
//!
//! ## Solid Archive Format
//!
//! ```text
//! [0..4]   "CPAS"            magic (4 bytes)
//! [4..8]   n_files: u32 LE
//! [per-file entry]:
//!   [2]  name_len: u16 LE
//!   [name_len]  UTF-8 file name
//!   [8]  orig_size: u64 LE
//! [rest]  CPGC single-stream payload (all file data concatenated, then compress())
//! ```

use anyhow::{anyhow, Result};

const MAGIC: &[u8; 4] = b"CPAS";

pub struct SolidArchive {
    pub files: Vec<SolidEntry>,
}

pub struct SolidEntry {
    pub name:   String,
    pub size:   u64,
    pub offset: u64, // byte offset in decompressed stream
}

impl SolidArchive {
    pub fn new() -> Self {
        Self { files: Vec::new() }
    }

    /// Compress `files` (slice of (name, data) pairs) into a solid archive.
    pub fn pack(files: &[(&str, &[u8])], level: u8) -> Result<Vec<u8>> {
        Self::pack_with_progress(files, level, |_, _| {})
    }

    /// Same as `pack` but forwards a progress callback to the codec.
    pub fn pack_with_progress(
        files: &[(&str, &[u8])],
        level: u8,
        on_progress: impl Fn(usize, usize),
    ) -> Result<Vec<u8>> {
        // Build file table
        let n = files.len() as u32;
        let mut table: Vec<u8> = Vec::new();
        table.extend_from_slice(MAGIC);
        table.extend_from_slice(&n.to_le_bytes());
        for &(name, data) in files {
            let name_bytes = name.as_bytes();
            table.extend_from_slice(&(name_bytes.len() as u16).to_le_bytes());
            table.extend_from_slice(name_bytes);
            table.extend_from_slice(&(data.len() as u64).to_le_bytes());
        }

        // Concatenate all file data and compress as one stream
        let total: usize = files.iter().map(|(_, d)| d.len()).sum();
        let mut combined = Vec::with_capacity(total);
        for &(_, data) in files {
            combined.extend_from_slice(data);
        }
        let payload = crate::codec::compress_with_progress(&combined, level, on_progress)?;

        let mut out = Vec::with_capacity(table.len() + payload.len());
        out.extend_from_slice(&table);
        out.extend_from_slice(&payload);
        Ok(out)
    }

    /// Decompress a solid archive, returning (name, data) pairs.
    pub fn unpack(input: &[u8]) -> Result<Vec<(String, Vec<u8>)>> {
        let (entries, payload_start) = Self::parse_table(input)?;

        let payload = &input[payload_start..];
        let combined = crate::codec::decompress(payload)?;

        let mut result = Vec::with_capacity(entries.len());
        let mut pos = 0usize;
        for (name, size) in entries {
            let end = pos + size as usize;
            if end > combined.len() {
                return Err(anyhow!("solid archive: data underrun for file {:?}", name));
            }
            result.push((name, combined[pos..end].to_vec()));
            pos = end;
        }
        Ok(result)
    }

    /// List file names and sizes without decompressing.
    pub fn list(input: &[u8]) -> Result<Vec<(String, u64)>> {
        let (entries, _) = Self::parse_table(input)?;
        Ok(entries)
    }

    // ------------------------------------------------------------------
    // Internal
    // ------------------------------------------------------------------

    fn parse_table(input: &[u8]) -> Result<(Vec<(String, u64)>, usize)> {
        if input.len() < 8 {
            return Err(anyhow!("solid archive too short"));
        }
        if &input[0..4] != MAGIC {
            return Err(anyhow!("not a CPGC solid archive (magic mismatch)"));
        }
        let n_files = u32::from_le_bytes(input[4..8].try_into().unwrap()) as usize;
        let mut pos = 8usize;
        let mut entries = Vec::with_capacity(n_files);
        for _ in 0..n_files {
            if pos + 2 > input.len() {
                return Err(anyhow!("solid archive: truncated file table"));
            }
            let name_len = u16::from_le_bytes(input[pos..pos + 2].try_into().unwrap()) as usize;
            pos += 2;
            if pos + name_len + 8 > input.len() {
                return Err(anyhow!("solid archive: truncated file entry"));
            }
            let name = String::from_utf8(input[pos..pos + name_len].to_vec())
                .map_err(|_| anyhow!("solid archive: invalid UTF-8 filename"))?;
            pos += name_len;
            let size = u64::from_le_bytes(input[pos..pos + 8].try_into().unwrap());
            pos += 8;
            entries.push((name, size));
        }
        Ok((entries, pos))
    }
}

