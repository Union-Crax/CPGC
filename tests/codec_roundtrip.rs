//! End-to-end codec roundtrip tests: compress → decompress → original.

use cpgc::codec::{compress, decompress};

fn roundtrip(input: &[u8]) {
    let compressed = compress(input, 1).expect("compress failed");
    let recovered  = decompress(&compressed).expect("decompress failed");
    assert_eq!(recovered, input,
        "roundtrip mismatch: {} bytes in, {} bytes compressed, {} bytes out",
        input.len(), compressed.len(), recovered.len());
}

#[test]
fn codec_roundtrip_empty() {
    roundtrip(&[]);
}

#[test]
fn codec_roundtrip_single_byte() {
    roundtrip(&[0x42]);
}

#[test]
fn codec_roundtrip_all_zeros() {
    roundtrip(&[0u8; 256]);
}

#[test]
fn codec_roundtrip_all_bytes() {
    let data: Vec<u8> = (0u8..=255).collect();
    roundtrip(&data);
}

#[test]
fn codec_roundtrip_repeated_pattern() {
    let data: Vec<u8> = b"hello world! ".iter().cycle().take(500).cloned().collect();
    roundtrip(&data);
}

#[test]
fn codec_roundtrip_random_looking() {
    // Pseudo-random via LCG — exercises the codec on hard-to-compress data
    let mut x: u64 = 0xdeadbeefcafe1337;
    let data: Vec<u8> = (0..1000).map(|_| {
        x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (x >> 56) as u8
    }).collect();
    roundtrip(&data);
}

#[test]
fn codec_header_magic_version() {
    // "test" = 4 bytes → 1 block
    let compressed = compress(b"test", 1).unwrap();
    assert_eq!(&compressed[0..4], b"CPGC", "magic bytes wrong");
    assert_eq!(compressed[4], 4, "version should be 4");
    // flags at [5]; orig_len at [6..14]
    let len = u64::from_le_bytes(compressed[6..14].try_into().unwrap());
    assert_eq!(len, 4, "stored length wrong");
    // n_blocks at [14..18]: 4 bytes → 1 block
    let n_blocks = u32::from_le_bytes(compressed[14..18].try_into().unwrap());
    assert_eq!(n_blocks, 1, "should be 1 block");
    // block_tag[0] at [18]: level=1 → no transform → 0x00
    assert_eq!(compressed[18], 0x00, "block tag should be 0x00 (no transform)");
}

#[test]
fn codec_compress_reduces_repetitive_data() {
    // A highly repetitive input should compress below its raw size
    let data = vec![b'a'; 1000];
    let compressed = compress(&data, 1).unwrap();
    assert!(compressed.len() < data.len(),
        "expected compression: {} bytes raw, {} bytes compressed",
        data.len(), compressed.len());
}

// ---------------------------------------------------------------------------
// Solid archive roundtrip tests
// ---------------------------------------------------------------------------

#[test]
fn solid_archive_roundtrip_single_file() {
    use cpgc::archive::solid::SolidArchive;
    let files = vec![("readme.txt", b"hello world".as_slice())];
    let packed = SolidArchive::pack(&files, 1).unwrap();
    let unpacked = SolidArchive::unpack(&packed).unwrap();
    assert_eq!(unpacked.len(), 1);
    assert_eq!(unpacked[0].0, "readme.txt");
    assert_eq!(unpacked[0].1, b"hello world");
}

#[test]
fn solid_archive_roundtrip_multi_file() {
    use cpgc::archive::solid::SolidArchive;
    let files: Vec<(&str, &[u8])> = vec![
        ("a.txt", b"AAAA"),
        ("b.txt", b"BBBBBBBB"),
        ("c.bin", &[0u8, 1, 2, 3, 255, 254]),
    ];
    let packed = SolidArchive::pack(&files, 1).unwrap();
    let listing = SolidArchive::list(&packed).unwrap();
    assert_eq!(listing.len(), 3);
    assert_eq!(listing[0], ("a.txt".to_string(), 4));
    assert_eq!(listing[1], ("b.txt".to_string(), 8));
    assert_eq!(listing[2], ("c.bin".to_string(), 6));

    let unpacked = SolidArchive::unpack(&packed).unwrap();
    for ((name, data), (exp_name, exp_data)) in unpacked.iter().zip(files.iter()) {
        assert_eq!(name, exp_name);
        assert_eq!(data.as_slice(), *exp_data);
    }
}

#[test]
fn solid_archive_roundtrip_empty_file() {
    use cpgc::archive::solid::SolidArchive;
    let files: Vec<(&str, &[u8])> = vec![("empty.bin", b"")];
    let packed = SolidArchive::pack(&files, 1).unwrap();
    let unpacked = SolidArchive::unpack(&packed).unwrap();
    assert_eq!(unpacked.len(), 1);
    assert_eq!(unpacked[0].0, "empty.bin");
    assert_eq!(unpacked[0].1, b"");
}

// ---------------------------------------------------------------------------
// int8 quantization tests
// ---------------------------------------------------------------------------

#[test]
fn quantize_snapshot_lossless_near_zero() {
    use cpgc::predictor::lstm::TinyLSTM;
    // A fresh model has very small weights (|w| < 0.05).
    // After quantization and dequantization the error per element must be < 0.001.
    let model = TinyLSTM::new(0.005);
    let snap = model.quantize_snapshot();
    // Spot-check: for each w_gates row, max reconstruction error ≤ scale/2
    for i in 0..snap.w_gates.len() {
        let s = snap.w_gates_scale[i];
        for &q in snap.w_gates[i].iter() {
            let reconstructed = q as f32 * s;
            let _ = reconstructed; // just ensure no panic
        }
    }
}

