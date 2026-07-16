// probe: encode a file as ONE segment, print rolling bpb every MiB
use std::env;
use std::fs;
fn main() {
    let args: Vec<String> = env::args().collect();
    let data = fs::read(&args[1]).unwrap();
    // memory profile: 0 = standard, 1 = big (level 7), 2 = extra (levels 8-9)
    let mem: u8 = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(0);
    cpgc::cm::probe_encode(&data, mem);
}
