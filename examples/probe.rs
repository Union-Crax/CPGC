// probe: encode a file as ONE segment, print rolling bpb every MiB
use std::env;
use std::fs;
fn main() {
    let args: Vec<String> = env::args().collect();
    let data = fs::read(&args[1]).unwrap();
    let big = args.get(2).map(|s| s == "big").unwrap_or(false);
    cpgc::cm::probe_encode(&data, big);
}
