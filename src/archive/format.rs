//! CPGC archive file format: header, region map, file table.

use std::io::{self, Read, Write};

pub const MAGIC: &[u8; 4] = b"CPGC";
pub const VERSION: u8 = 0x03;

/// Flags byte in the file header.
pub mod flags {
    pub const IS_ARCHIVE:    u8 = 0x01;
    pub const SOLID_MODE:    u8 = 0x02;
    pub const TRANSFORM_USED: u8 = 0x04;
}

/// File header: exactly 40 bytes.
#[derive(Debug, Clone)]
pub struct FileHeader {
    pub version:           u8,
    pub level:             u8,
    pub lstm_hidden_log2:  u8, // 0 = disabled, 5 = 32, 6 = 64, 7 = 128, 8 = 256
    pub flags:             u8,
    pub uncompressed_size: u64,
    pub compressed_size:   u64,
    pub crc32:             u32,
    pub xxhash3_lo:        u32,
    pub region_map_len:    u32,
    pub file_table_offset: u32,
}

impl FileHeader {
    pub fn write<W: Write>(&self, w: &mut W) -> io::Result<()> {
        w.write_all(MAGIC)?;
        w.write_all(&[self.version, self.level, self.lstm_hidden_log2, self.flags])?;
        w.write_all(&self.uncompressed_size.to_le_bytes())?;
        w.write_all(&self.compressed_size.to_le_bytes())?;
        w.write_all(&self.crc32.to_le_bytes())?;
        w.write_all(&self.xxhash3_lo.to_le_bytes())?;
        w.write_all(&self.region_map_len.to_le_bytes())?;
        w.write_all(&self.file_table_offset.to_le_bytes())?;
        Ok(())
    }

    pub fn read<R: Read>(r: &mut R) -> io::Result<Self> {
        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "bad CPGC magic"));
        }
        let mut buf4 = [0u8; 4];
        r.read_exact(&mut buf4)?;
        let (version, level, lstm_hidden_log2, flags) = (buf4[0], buf4[1], buf4[2], buf4[3]);

        let mut buf8 = [0u8; 8];
        r.read_exact(&mut buf8)?; let uncompressed_size = u64::from_le_bytes(buf8);
        r.read_exact(&mut buf8)?; let compressed_size   = u64::from_le_bytes(buf8);

        let mut buf4a = [0u8; 4];
        r.read_exact(&mut buf4a)?; let crc32         = u32::from_le_bytes(buf4a);
        r.read_exact(&mut buf4a)?; let xxhash3_lo    = u32::from_le_bytes(buf4a);
        r.read_exact(&mut buf4a)?; let region_map_len = u32::from_le_bytes(buf4a);
        r.read_exact(&mut buf4a)?; let file_table_offset = u32::from_le_bytes(buf4a);

        Ok(Self { version, level, lstm_hidden_log2, flags,
                  uncompressed_size, compressed_size, crc32, xxhash3_lo,
                  region_map_len, file_table_offset })
    }
}

/// Region map entry: 6 bytes each.
#[derive(Debug, Clone, Copy)]
pub struct RegionEntry {
    pub offset:  u32,
    pub preproc: u8, // flags as defined in plan
    pub _pad:    u8,
}

impl RegionEntry {
    pub fn write<W: Write>(&self, w: &mut W) -> io::Result<()> {
        w.write_all(&self.offset.to_le_bytes())?;
        w.write_all(&[self.preproc, self._pad])?;
        Ok(())
    }

    pub fn read<R: Read>(r: &mut R) -> io::Result<Self> {
        let mut b4 = [0u8; 4];
        r.read_exact(&mut b4)?;
        let mut b2 = [0u8; 2];
        r.read_exact(&mut b2)?;
        Ok(Self { offset: u32::from_le_bytes(b4), preproc: b2[0], _pad: b2[1] })
    }
}
