use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser)]
#[command(name = "cpgc", about = "Contextual Predictive Graph Compression")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Compress a file
    Compress {
        input: PathBuf,
        output: PathBuf,
        #[arg(short, long, default_value_t = 5)]
        level: u8,
    },
    /// Decompress a file
    Decompress {
        input: PathBuf,
        output: PathBuf,
    },
    /// List archive contents (placeholder — solid archive not yet implemented)
    List {
        archive: PathBuf,
    },
    /// Benchmark on a corpus directory
    Bench {
        corpus_dir: PathBuf,
    },
    /// Show compressed file info
    Info {
        archive: PathBuf,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Compress { input, output, level } => cmd_compress(&input, &output, level),
        Commands::Decompress { input, output }       => cmd_decompress(&input, &output),
        Commands::List { archive }                   => cmd_list(&archive),
        Commands::Bench { corpus_dir }               => cmd_bench(&corpus_dir),
        Commands::Info { archive }                   => cmd_info(&archive),
    }
}

fn cmd_compress(input: &PathBuf, output: &PathBuf, level: u8) -> Result<()> {
    // If input is a directory, create a solid multi-file archive.
    if input.is_dir() {
        use cpgc::archive::solid::SolidArchive;
        use walkdir::WalkDir;
        let mut files: Vec<(String, Vec<u8>)> = Vec::new();
        for entry in WalkDir::new(input).sort_by_file_name() {
            let entry = entry.with_context(|| format!("walking {:?}", input))?;
            if !entry.file_type().is_file() { continue; }
            let rel = entry.path().strip_prefix(input)
                .unwrap_or(entry.path())
                .to_string_lossy()
                .replace('\\', "/");
            let data = std::fs::read(entry.path())
                .with_context(|| format!("reading {:?}", entry.path()))?;
            files.push((rel, data));
        }
        let pairs: Vec<(&str, &[u8])> = files.iter().map(|(n, d)| (n.as_str(), d.as_slice())).collect();
        let t0 = Instant::now();
        let packed = SolidArchive::pack(&pairs, level)?;
        let elapsed = t0.elapsed().as_secs_f64();
        std::fs::write(output, &packed)
            .with_context(|| format!("writing {:?}", output))?;
        let total_raw: usize = files.iter().map(|(_, d)| d.len()).sum();
        println!(
            "{:?} ({} files) → {:?}\n  {:>10} bytes → {:>10} bytes  ({:.3} ratio)\n  {:.3} MB/s  ({:.2}s)",
            input, files.len(), output,
            total_raw, packed.len(),
            packed.len() as f64 / total_raw.max(1) as f64,
            total_raw as f64 / elapsed / 1_000_000.0, elapsed,
        );
        return Ok(());
    }

    let data = std::fs::read(input)
        .with_context(|| format!("reading {:?}", input))?;

    let t0 = Instant::now();
    let compressed = cpgc::codec::compress(&data, level)?;
    let elapsed = t0.elapsed().as_secs_f64();

    std::fs::write(output, &compressed)
        .with_context(|| format!("writing {:?}", output))?;

    let ratio = compressed.len() as f64 / data.len().max(1) as f64;
    let mb_s  = data.len() as f64 / elapsed / 1_000_000.0;
    println!(
        "{:?} → {:?}\n  {:>10} bytes → {:>10} bytes  ({:.3} ratio)\n  {:.3} MB/s  ({:.2}s)",
        input, output, data.len(), compressed.len(), ratio, mb_s, elapsed
    );
    Ok(())
}

fn cmd_decompress(input: &PathBuf, output: &PathBuf) -> Result<()> {
    let data = std::fs::read(input)
        .with_context(|| format!("reading {:?}", input))?;

    let t0 = Instant::now();
    let recovered = cpgc::codec::decompress(&data)?;
    let elapsed = t0.elapsed().as_secs_f64();

    std::fs::write(output, &recovered)
        .with_context(|| format!("writing {:?}", output))?;

    let mb_s = recovered.len() as f64 / elapsed / 1_000_000.0;
    println!(
        "{:?} → {:?}\n  {:>10} bytes recovered  ({:.3} MB/s, {:.2}s)",
        input, output, recovered.len(), mb_s, elapsed
    );
    Ok(())
}

fn cmd_info(archive: &PathBuf) -> Result<()> {
    let data = std::fs::read(archive)
        .with_context(|| format!("reading {:?}", archive))?;

    // Check for solid archive (CPAS magic)
    if data.starts_with(b"CPAS") {
        use cpgc::archive::solid::SolidArchive;
        let entries = SolidArchive::list(&data)?;
        println!("CPGC solid archive: {:?}  ({} files)", archive, entries.len());
        println!("  compressed: {} bytes", data.len());
        for (name, size) in &entries {
            println!("    {:>12} B  {}", size, name);
        }
        return Ok(());
    }

    // VERSION 2 single-file header layout:
    //   [0..4]   magic "CPGC"
    //   [4]      version
    //   [5]      flags
    //   [6..14]  orig_len u64 LE
    //   [14..18] n_blocks u32 LE
    //   [18+n_blocks..22+n_blocks] passthrough_len u32 LE
    const MIN: usize = 22;
    if data.len() < MIN {
        bail!("file too small to be a CPGC archive");
    }
    if &data[0..4] != b"CPGC" {
        bail!("not a CPGC archive (magic mismatch)");
    }
    let version   = data[4];
    let flags     = data[5];
    let orig_len  = u64::from_le_bytes(data[6..14].try_into().unwrap());
    let n_blocks  = u32::from_le_bytes(data[14..18].try_into().unwrap()) as usize;
    let pt_len_off = 18 + n_blocks;
    let passthrough_len = if data.len() >= pt_len_off + 4 {
        u32::from_le_bytes(data[pt_len_off..pt_len_off + 4].try_into().unwrap()) as usize
    } else { 0 };
    let ans_payload = data.len().saturating_sub(pt_len_off + 4 + passthrough_len);
    let ratio = data.len() as f64 / orig_len.max(1) as f64;
    let bpb   = data.len() as f64 * 8.0 / orig_len.max(1) as f64;

    println!("CPGC archive: {:?}", archive);
    println!("  version:        {}", version);
    println!("  flags:          0x{:02x}  ({}{})", flags,
        if flags & 1 != 0 { "passthrough " } else { "" },
        if flags & 2 != 0 { "transforms" } else { "" });
    println!("  original size:  {} bytes", orig_len);
    println!("  compressed:     {} bytes  (ANS payload: {} bytes, passthrough: {} bytes)",
        data.len(), ans_payload, passthrough_len);
    println!("  blocks:         {} × {}B", n_blocks, cpgc::analyzer::classifier::WINDOW_SIZE);
    println!("  ratio:          {:.4}", ratio);
    println!("  bits/byte:      {:.4}", bpb);
    Ok(())
}

fn cmd_list(archive: &PathBuf) -> Result<()> {
    let data = std::fs::read(archive)
        .with_context(|| format!("reading {:?}", archive))?;

    if data.starts_with(b"CPAS") {
        use cpgc::archive::solid::SolidArchive;
        let entries = SolidArchive::list(&data)?;
        println!("{:>12}  {}", "size (B)", "name");
        println!("{}", "-".repeat(60));
        for (name, size) in &entries {
            println!("{:>12}  {}", size, name);
        }
        println!("{} files", entries.len());
    } else {
        println!("Note: single-file CPGC archive (not a solid multi-file archive). Showing info:");
        cmd_info(archive)?;
    }
    Ok(())
}

fn cmd_bench(corpus_dir: &PathBuf) -> Result<()> {
    use std::fs;

    let entries = fs::read_dir(corpus_dir)
        .with_context(|| format!("reading directory {:?}", corpus_dir))?;

    let mut files: Vec<PathBuf> = entries
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_file())
        .collect();
    files.sort();

    if files.is_empty() {
        bail!("no files found in {:?}", corpus_dir);
    }

    println!("{:<40}  {:>10}  {:>10}  {:>7}  {:>8}  {:>8}",
        "file", "orig(B)", "comp(B)", "ratio", "bpb", "MB/s");
    println!("{}", "-".repeat(90));

    let mut total_orig = 0usize;
    let mut total_comp = 0usize;

    for path in &files {
        let data = match fs::read(path) {
            Ok(d) => d,
            Err(e) => { eprintln!("skip {:?}: {}", path, e); continue; }
        };
        let t0 = Instant::now();
        let comp = match cpgc::codec::compress(&data, 5) {
            Ok(c) => c,
            Err(e) => { eprintln!("error {:?}: {}", path, e); continue; }
        };
        let elapsed = t0.elapsed().as_secs_f64();

        let name = path.file_name().unwrap_or_default().to_string_lossy();
        let ratio = comp.len() as f64 / data.len().max(1) as f64;
        let bpb   = comp.len() as f64 * 8.0 / data.len().max(1) as f64;
        let mb_s  = data.len() as f64 / elapsed / 1_000_000.0;

        println!("{:<40}  {:>10}  {:>10}  {:>7.4}  {:>8.4}  {:>8.3}",
            &name[..name.len().min(40)], data.len(), comp.len(), ratio, bpb, mb_s);

        total_orig += data.len();
        total_comp += comp.len();
    }

    println!("{}", "-".repeat(90));
    let total_ratio = total_comp as f64 / total_orig.max(1) as f64;
    let total_bpb   = total_comp as f64 * 8.0 / total_orig.max(1) as f64;
    println!("{:<40}  {:>10}  {:>10}  {:>7.4}  {:>8.4}",
        "TOTAL", total_orig, total_comp, total_ratio, total_bpb);
    Ok(())
}

