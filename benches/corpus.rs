use criterion::{criterion_group, criterion_main, Criterion, Throughput};

// ---------------------------------------------------------------------------
// LSTM forward+update throughput
// ---------------------------------------------------------------------------

fn bench_lstm(c: &mut Criterion) {
    use cpgc::predictor::lstm::TinyLSTM;

    let data: Vec<u8> = (0u8..=255).cycle().take(10_000).collect();
    let mut group = c.benchmark_group("lstm");
    group.throughput(Throughput::Bytes(data.len() as u64));

    group.bench_function("forward_update_10k", |b| {
        b.iter(|| {
            let mut model = TinyLSTM::new(0.005);
            for &byte in &data {
                let _probs = model.forward(byte);
                model.update(byte);
            }
        });
    });
    group.finish();
}

// ---------------------------------------------------------------------------
// ANS encode+decode throughput
// ---------------------------------------------------------------------------

fn bench_ans(c: &mut Criterion) {
    use cpgc::ans::encode::AnsEncoder;
    use cpgc::ans::decode::AnsDecoder;

    let data: Vec<u8> = (0u8..=255).cycle().take(4_096).collect();
    let uniform = [1.0f32 / 256.0; 256];

    let mut group = c.benchmark_group("ans");
    group.throughput(Throughput::Bytes(data.len() as u64));

    group.bench_function("encode_4k", |b| {
        b.iter(|| {
            let mut enc = AnsEncoder::new(&uniform);
            for &byte in &data { enc.encode(byte); }
            enc.finish()
        });
    });

    // Pre-encode payload for decode bench
    let payload = {
        let mut enc = AnsEncoder::new(&uniform);
        for &byte in &data { enc.encode(byte); }
        enc.finish()
    };

    group.bench_function("decode_4k", |b| {
        b.iter(|| {
            let mut dec = AnsDecoder::new(&payload, &uniform).unwrap();
            let mut out = Vec::with_capacity(data.len());
            for _ in 0..data.len() { out.push(dec.decode().unwrap()); }
            out
        });
    });
    group.finish();
}

// ---------------------------------------------------------------------------
// End-to-end codec throughput (uses enwik8 corpus if available)
// ---------------------------------------------------------------------------

fn bench_codec(c: &mut Criterion) {
    // Use first 10 KB of enwik8 if available; fall back to synthetic data.
    let data: Vec<u8> = std::fs::read("corpus/enwik8")
        .map(|d| d.into_iter().take(10_240).collect())
        .unwrap_or_else(|_| {
            let mut x: u64 = 0xdeadbeef_cafe1234;
            (0..10_240).map(|_| {
                x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                (x >> 56) as u8
            }).collect()
        });

    let mut group = c.benchmark_group("codec");
    group.throughput(Throughput::Bytes(data.len() as u64));
    // Codec is slow (online LSTM); limit sample count to avoid very long bench runs.
    group.sample_size(10);

    group.bench_function("compress_10k", |b| {
        b.iter(|| cpgc::codec::compress(&data, 1).unwrap())
    });

    let compressed = cpgc::codec::compress(&data, 1).unwrap();
    group.bench_function("decompress_10k", |b| {
        b.iter(|| cpgc::codec::decompress(&compressed).unwrap())
    });

    group.finish();
}

criterion_group!(benches, bench_lstm, bench_ans, bench_codec);
criterion_main!(benches);

