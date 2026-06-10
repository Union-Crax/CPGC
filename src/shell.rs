//! Windows Explorer shell integration — the 7-Zip-style right-click menu.
//!
//! `cpgc register` writes a small set of keys under `HKCU\Software\Classes`
//! (the per-user class store, so **no administrator rights are required** and
//! nothing system-wide is touched). `cpgc unregister` removes them again.
//!
//! What it installs:
//!
//! * **Compress with CPGC** on every file and folder — runs `cpgc compress`
//!   on the item, producing a `.cpgc` (single file) or `.cpas` (folder)
//!   archive next to it, in a console window that shows progress.
//! * A file association for `.cpgc` / `.cpas` archives with these verbs:
//!   * **Open with CPGC** (the default action / double-click) opens the
//!     archive in the native GUI to browse and extract members.
//!   * **Extract here** runs `cpgc decompress` next to the archive.
//!   * **Test CPGC archive** runs `cpgc verify`.
//!
//! The console verbs are wrapped in `cmd /c "… & pause"` so the window stays
//! open with the result instead of flashing past.

use anyhow::{Context, Result};
use winreg::enums::HKEY_CURRENT_USER;
use winreg::RegKey;

/// Programmatic identifier our archive extensions point at.
const PROGID: &str = "CPGC.Archive";
const CLASSES: &str = r"Software\Classes";

/// Absolute path to the running `cpgc` executable, with Windows separators.
fn exe_path() -> Result<String> {
    let p = std::env::current_exe().context("locating the cpgc executable")?;
    Ok(p.to_string_lossy().replace('/', "\\"))
}

/// Install the right-click menu entries for the current user.
pub fn register() -> Result<()> {
    let exe = exe_path()?;
    let hkcu = RegKey::predef(HKEY_CURRENT_USER);

    // --- ProgID describing a CPGC archive --------------------------------
    let (k, _) = hkcu.create_subkey(format!(r"{CLASSES}\{PROGID}"))?;
    k.set_value("", &"CPGC Archive")?;
    let (ki, _) = hkcu.create_subkey(format!(r"{CLASSES}\{PROGID}\DefaultIcon"))?;
    ki.set_value("", &format!("{exe},0"))?;

    // Double-click / default verb → open the archive in the GUI.
    set_verb(
        &hkcu,
        &format!(r"{CLASSES}\{PROGID}\shell\open"),
        "Open with CPGC",
        &format!("\"{exe}\" gui --open \"%1\""),
        &exe,
    )?;
    // Extract here → decompress beside the archive (console with progress).
    set_verb(
        &hkcu,
        &format!(r"{CLASSES}\{PROGID}\shell\extract"),
        "Extract here",
        &format!("cmd /c \"\"{exe}\" decompress \"%1\" & pause\""),
        &exe,
    )?;
    // Test → verify the archive decodes and its checksum matches.
    set_verb(
        &hkcu,
        &format!(r"{CLASSES}\{PROGID}\shell\test"),
        "Test CPGC archive",
        &format!("cmd /c \"\"{exe}\" verify \"%1\" & pause\""),
        &exe,
    )?;

    // --- associate the archive extensions with the ProgID ----------------
    for ext in [".cpgc", ".cpas"] {
        let (kx, _) = hkcu.create_subkey(format!(r"{CLASSES}\{ext}"))?;
        kx.set_value("", &PROGID)?;
    }

    // --- "Compress with CPGC" on every file and every folder -------------
    for base in ["*", "Directory"] {
        set_verb(
            &hkcu,
            &format!(r"{CLASSES}\{base}\shell\CPGCCompress"),
            "Compress with CPGC",
            &format!("cmd /c \"\"{exe}\" compress \"%1\" & pause\""),
            &exe,
        )?;
    }

    Ok(())
}

/// Remove every key `register` created (best-effort; missing keys are fine).
pub fn unregister() -> Result<()> {
    let hkcu = RegKey::predef(HKEY_CURRENT_USER);
    let _ = hkcu.delete_subkey_all(format!(r"{CLASSES}\{PROGID}"));
    let _ = hkcu.delete_subkey_all(format!(r"{CLASSES}\*\shell\CPGCCompress"));
    let _ = hkcu.delete_subkey_all(format!(r"{CLASSES}\Directory\shell\CPGCCompress"));
    // Only drop an extension association that still points at our ProgID, so we
    // never clobber an association the user has since pointed elsewhere.
    for ext in [".cpgc", ".cpas"] {
        if let Ok(k) = hkcu.open_subkey(format!(r"{CLASSES}\{ext}")) {
            let cur: std::io::Result<String> = k.get_value("");
            if cur.ok().as_deref() == Some(PROGID) {
                let _ = hkcu.delete_subkey_all(format!(r"{CLASSES}\{ext}"));
            }
        }
    }
    Ok(())
}

/// Create `path` with a display `label` (and optional icon) plus a
/// `path\command` subkey holding `command`.
fn set_verb(hkcu: &RegKey, path: &str, label: &str, command: &str, icon: &str) -> Result<()> {
    let (k, _) = hkcu.create_subkey(path)?;
    k.set_value("", &label)?;
    k.set_value("Icon", &format!("{icon},0"))?;
    let (kc, _) = hkcu.create_subkey(format!(r"{path}\command"))?;
    kc.set_value("", &command)?;
    Ok(())
}
