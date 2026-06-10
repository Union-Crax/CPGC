use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let (dir, archive) = if args.len() >= 2 {
        let p = PathBuf::from(&args[1]);
        let lower = args[1].to_lowercase();
        if lower.ends_with(".cpgc") || lower.ends_with(".cpas") {
            // Launched by Explorer "Open with": start in the archive's folder and open it.
            let start = p.parent()
                .map(|d| d.to_path_buf())
                .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
            (start, Some(p))
        } else {
            (p, None)
        }
    } else {
        let dir = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        (dir, None)
    };
    cpgc::gui::run(dir, archive)
}
