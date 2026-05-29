//! `cpgc-gui` — the native desktop application (7-Zip-style file manager).
//!
//! This is a separate binary from the `cpgc` command-line tool so the CLI can
//! be shipped without the GUI's windowing dependencies. The first optional
//! argument is the folder to open on start-up.

use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    let dir = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
    cpgc::gui::run(dir)
}
