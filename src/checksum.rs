pub fn crc32(data: &[u8]) -> u32 {
    crc32fast::hash(data)
}

pub fn xxhash3(data: &[u8]) -> u64 {
    xxhash_rust::xxh3::xxh3_64(data)
}
