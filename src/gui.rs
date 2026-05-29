//! Native desktop GUI — a 7-Zip-style file manager built with egui/eframe.
//!
//! `cpgc gui` opens a real OS window (no browser): browse folders, tick files
//! or directories, pick a compression level, and compress them into a `.cpgc`
//! single-file archive or a `.cpas` solid multi-file archive. Select an archive
//! to extract or verify it. Long operations run on a background thread with a
//! live progress bar so the UI never freezes.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, Sender};
use std::time::Instant;

use anyhow::{anyhow, Result};
use eframe::egui;

use crate::archive::solid::SolidArchive;
use crate::codec;

/// Launch the GUI. `start_dir` is the initial directory shown.
pub fn run(start_dir: PathBuf) -> Result<()> {
    let start_dir = start_dir
        .canonicalize()
        .unwrap_or_else(|_| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([960.0, 620.0])
            .with_min_inner_size([560.0, 360.0])
            .with_title("CPGC — Compressor"),
        ..Default::default()
    };

    eframe::run_native(
        "CPGC",
        options,
        Box::new(move |_cc| Box::new(CpgcApp::new(start_dir))),
    )
    .map_err(|e| anyhow!("GUI failed to start: {e}"))
}

// ---------------------------------------------------------------------------
// App state
// ---------------------------------------------------------------------------

struct Entry {
    path: PathBuf,
    name: String,
    is_dir: bool,
    is_archive: bool,
    size: u64,
}

#[derive(PartialEq)]
enum StatusKind {
    Info,
    Ok,
    Err,
}

/// Messages from a background worker to the UI thread.
enum JobMsg {
    Progress(f32),
    Done(std::result::Result<String, String>),
}

struct Job {
    rx: Receiver<JobMsg>,
    label: String,
    progress: f32,
    started: Instant,
}

struct CpgcApp {
    cwd: PathBuf,
    entries: Vec<Entry>,
    selected: BTreeSet<PathBuf>,
    level: u8,
    out_name: String,
    extract_dest: String,
    status: String,
    status_kind: StatusKind,
    job: Option<Job>,
}

impl CpgcApp {
    fn new(cwd: PathBuf) -> Self {
        let mut app = Self {
            cwd,
            entries: Vec::new(),
            selected: BTreeSet::new(),
            level: 5,
            out_name: "archive.cpgc".to_string(),
            extract_dest: String::new(),
            status: "Ready. Tick items to compress, or pick an archive to extract.".to_string(),
            status_kind: StatusKind::Info,
            job: None,
        };
        app.refresh();
        app
    }

    fn refresh(&mut self) {
        self.entries.clear();
        if let Ok(rd) = std::fs::read_dir(&self.cwd) {
            for e in rd.flatten() {
                let path = e.path();
                let name = e.file_name().to_string_lossy().to_string();
                let (is_dir, size) = match e.metadata() {
                    Ok(m) => (m.is_dir(), m.len()),
                    Err(_) => (false, 0),
                };
                let lower = name.to_lowercase();
                let is_archive = lower.ends_with(".cpgc") || lower.ends_with(".cpas");
                self.entries.push(Entry { path, name, is_dir, is_archive, size });
            }
        }
        self.entries.sort_by(|a, b| {
            b.is_dir
                .cmp(&a.is_dir)
                .then(a.name.to_lowercase().cmp(&b.name.to_lowercase()))
        });
        // Drop selections that are no longer present.
        let present: BTreeSet<PathBuf> = self.entries.iter().map(|e| e.path.clone()).collect();
        self.selected.retain(|p| present.contains(p));
    }

    fn navigate(&mut self, dir: PathBuf) {
        if let Ok(c) = dir.canonicalize() {
            self.cwd = c;
            self.selected.clear();
            self.refresh();
        }
    }

    fn set_status(&mut self, kind: StatusKind, msg: impl Into<String>) {
        self.status_kind = kind;
        self.status = msg.into();
    }

    fn busy(&self) -> bool {
        self.job.is_some()
    }

    // --- background jobs ------------------------------------------------

    fn poll_job(&mut self) {
        let mut finished: Option<std::result::Result<String, String>> = None;
        if let Some(job) = &mut self.job {
            while let Ok(msg) = job.rx.try_recv() {
                match msg {
                    JobMsg::Progress(p) => job.progress = p,
                    JobMsg::Done(res) => finished = Some(res),
                }
            }
        }
        if let Some(res) = finished {
            match res {
                Ok(m) => self.set_status(StatusKind::Ok, m),
                Err(e) => self.set_status(StatusKind::Err, e),
            }
            self.job = None;
            self.refresh();
        }
    }

    fn start_compress(&mut self) {
        if self.busy() || self.selected.is_empty() {
            return;
        }
        let mut out = self.out_name.trim().to_string();
        if out.is_empty() {
            out = "archive.cpgc".to_string();
        }
        if !out.to_lowercase().ends_with(".cpgc") && !out.to_lowercase().ends_with(".cpas") {
            out.push_str(".cpgc");
        }
        let output = self.cwd.join(&out);
        let inputs: Vec<PathBuf> = self.selected.iter().cloned().collect();
        let level = self.level;

        let (tx, rx) = std::sync::mpsc::channel();
        let label = format!("Compressing {} item(s) → {}", inputs.len(), out);
        self.set_status(StatusKind::Info, label.clone());
        spawn_compress(inputs, output, level, tx);
        self.job = Some(Job { rx, label, progress: 0.0, started: Instant::now() });
    }

    fn start_extract(&mut self, archive: PathBuf) {
        if self.busy() {
            return;
        }
        let base = archive
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "extracted".to_string());
        let dest = if self.extract_dest.trim().is_empty() {
            self.cwd.join(format!("{base}_extracted"))
        } else {
            self.cwd.join(self.extract_dest.trim())
        };
        let (tx, rx) = std::sync::mpsc::channel();
        let label = format!("Extracting {}", file_name(&archive));
        self.set_status(StatusKind::Info, label.clone());
        spawn_extract(archive, dest, tx);
        self.job = Some(Job { rx, label, progress: 0.0, started: Instant::now() });
    }

    fn start_verify(&mut self, archive: PathBuf) {
        if self.busy() {
            return;
        }
        let (tx, rx) = std::sync::mpsc::channel();
        let label = format!("Verifying {}", file_name(&archive));
        self.set_status(StatusKind::Info, label.clone());
        spawn_verify(archive, tx);
        self.job = Some(Job { rx, label, progress: 0.0, started: Instant::now() });
    }
}

// ---------------------------------------------------------------------------
// UI
// ---------------------------------------------------------------------------

impl eframe::App for CpgcApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll_job();
        if self.busy() {
            ctx.request_repaint(); // keep the progress bar animating
        }

        self.top_bar(ctx);
        self.bottom_bar(ctx);
        self.file_list(ctx);
    }
}

impl CpgcApp {
    fn top_bar(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::top("toolbar").show(ctx, |ui| {
            ui.add_space(4.0);
            ui.horizontal_wrapped(|ui| {
                ui.heading("📦 CPGC");
                ui.separator();

                let busy = self.busy();
                ui.add_enabled_ui(!busy, |ui| {
                    if ui.button("⬆ Up").clicked() {
                        if let Some(parent) = self.cwd.parent() {
                            self.navigate(parent.to_path_buf());
                        }
                    }
                });

                ui.label(format!("{} selected", self.selected.len()));
                ui.separator();

                ui.label("Level");
                egui::ComboBox::from_id_source("level")
                    .selected_text(level_label(self.level))
                    .show_ui(ui, |ui| {
                        for lvl in 1u8..=9 {
                            ui.selectable_value(&mut self.level, lvl, level_label(lvl));
                        }
                    });

                ui.separator();
                ui.label("Output");
                ui.add(egui::TextEdit::singleline(&mut self.out_name).desired_width(160.0));

                ui.add_enabled_ui(!busy && !self.selected.is_empty(), |ui| {
                    if ui.button("🗜 Compress").clicked() {
                        self.start_compress();
                    }
                });
            });
            ui.add_space(2.0);
            ui.horizontal(|ui| {
                ui.label("📂");
                ui.monospace(self.cwd.to_string_lossy());
            });
            ui.add_space(4.0);
        });
    }

    fn bottom_bar(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::bottom("status").show(ctx, |ui| {
            ui.add_space(4.0);
            if let Some(job) = &self.job {
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.label(&job.label);
                });
                let p = if job.progress > 0.0 { job.progress } else { 0.02 };
                ui.add(egui::ProgressBar::new(p).show_percentage().animate(true));
            } else {
                let color = match self.status_kind {
                    StatusKind::Ok => egui::Color32::from_rgb(0x3f, 0xbf, 0x6f),
                    StatusKind::Err => egui::Color32::from_rgb(0xff, 0x6b, 0x6b),
                    StatusKind::Info => ui.visuals().text_color(),
                };
                ui.colored_label(color, &self.status);
            }
            ui.add_space(4.0);
        });
    }

    fn file_list(&mut self, ctx: &egui::Context) {
        // Actions deferred until after the immutable iteration over entries.
        let mut nav_to: Option<PathBuf> = None;
        let mut toggle: Option<PathBuf> = None;
        let mut extract: Option<PathBuf> = None;
        let mut verify: Option<PathBuf> = None;
        let busy = self.busy();

        egui::CentralPanel::default().show(ctx, |ui| {
            egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                egui::Grid::new("files")
                    .num_columns(4)
                    .striped(true)
                    .spacing([12.0, 6.0])
                    .show(ui, |ui| {
                        ui.strong("");
                        ui.strong("Name");
                        ui.strong("Size");
                        ui.strong("Action");
                        ui.end_row();

                        for e in &self.entries {
                            // Selection checkbox (files and folders both selectable).
                            let mut sel = self.selected.contains(&e.path);
                            if ui.add_enabled(!busy, egui::Checkbox::without_text(&mut sel)).changed() {
                                toggle = Some(e.path.clone());
                            }

                            // Name — folders navigate, files toggle selection.
                            let icon = if e.is_dir {
                                "📁"
                            } else if e.is_archive {
                                "🗜"
                            } else {
                                "📄"
                            };
                            let label = format!("{icon} {}", e.name);
                            if e.is_dir {
                                if ui.add_enabled(!busy, egui::Button::new(label).frame(false)).clicked() {
                                    nav_to = Some(e.path.clone());
                                }
                            } else if ui.add(egui::SelectableLabel::new(sel, label)).clicked() {
                                toggle = Some(e.path.clone());
                            }

                            // Size.
                            if e.is_dir {
                                ui.label("—");
                            } else {
                                ui.monospace(human_size(e.size));
                            }

                            // Per-row actions for archives.
                            ui.horizontal(|ui| {
                                if e.is_archive {
                                    ui.add_enabled_ui(!busy, |ui| {
                                        if ui.button("Extract").clicked() {
                                            extract = Some(e.path.clone());
                                        }
                                        if ui.button("Verify").clicked() {
                                            verify = Some(e.path.clone());
                                        }
                                    });
                                }
                            });
                            ui.end_row();
                        }
                    });
            });
        });

        if let Some(p) = toggle {
            if !self.selected.remove(&p) {
                self.selected.insert(p);
            }
        }
        if let Some(p) = nav_to {
            self.navigate(p);
        }
        if let Some(p) = extract {
            self.start_extract(p);
        }
        if let Some(p) = verify {
            self.start_verify(p);
        }
    }
}

// ---------------------------------------------------------------------------
// Background workers
// ---------------------------------------------------------------------------

fn spawn_compress(inputs: Vec<PathBuf>, output: PathBuf, level: u8, tx: Sender<JobMsg>) {
    std::thread::spawn(move || {
        let res = do_compress(&inputs, &output, level, &tx).map_err(|e| format!("{e:#}"));
        let _ = tx.send(JobMsg::Done(res));
    });
}

fn do_compress(
    inputs: &[PathBuf],
    output: &Path,
    level: u8,
    tx: &Sender<JobMsg>,
) -> Result<String> {
    // Gather (relative_name, bytes) for every input file.
    let mut files: Vec<(String, Vec<u8>)> = Vec::new();
    for inp in inputs {
        if inp.is_dir() {
            let base = inp.parent().unwrap_or(inp);
            for entry in walkdir::WalkDir::new(inp).sort_by_file_name() {
                let entry = entry?;
                if !entry.file_type().is_file() {
                    continue;
                }
                let rel = entry
                    .path()
                    .strip_prefix(base)
                    .unwrap_or(entry.path())
                    .to_string_lossy()
                    .replace('\\', "/");
                files.push((rel, std::fs::read(entry.path())?));
            }
        } else {
            files.push((file_name(inp), std::fs::read(inp)?));
        }
    }
    if files.is_empty() {
        return Err(anyhow!("nothing to compress"));
    }
    let total_raw: usize = files.iter().map(|(_, d)| d.len()).sum();

    let progress = |done: usize, total: usize| {
        let _ = tx.send(JobMsg::Progress(done as f32 / total.max(1) as f32));
    };

    let packed = if files.len() == 1 {
        codec::compress_with_progress(&files[0].1, level, progress)?
    } else {
        let pairs: Vec<(&str, &[u8])> =
            files.iter().map(|(n, d)| (n.as_str(), d.as_slice())).collect();
        SolidArchive::pack_with_progress(&pairs, level, progress)?
    };
    std::fs::write(output, &packed)?;

    let pct = packed.len() as f64 / total_raw.max(1) as f64 * 100.0;
    Ok(format!(
        "✔ {} — {} → {} ({:.1}% of original, {} file(s))",
        file_name(output),
        human_size(total_raw as u64),
        human_size(packed.len() as u64),
        pct,
        files.len()
    ))
}

fn spawn_extract(archive: PathBuf, dest: PathBuf, tx: Sender<JobMsg>) {
    std::thread::spawn(move || {
        let res = do_extract(&archive, &dest).map_err(|e| format!("{e:#}"));
        let _ = tx.send(JobMsg::Done(res));
    });
}

fn do_extract(archive: &Path, dest: &Path) -> Result<String> {
    let data = std::fs::read(archive)?;
    std::fs::create_dir_all(dest)?;
    if data.starts_with(b"CPAS") {
        let files = SolidArchive::unpack(&data)?;
        for (name, bytes) in &files {
            let safe = sanitize_rel(name);
            let out = dest.join(&safe);
            if let Some(parent) = out.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&out, bytes)?;
        }
        Ok(format!(
            "✔ extracted {} file(s) → {}",
            files.len(),
            file_name(dest)
        ))
    } else {
        let recovered = codec::decompress(&data)?;
        let stem = archive
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "recovered".to_string());
        std::fs::write(dest.join(&stem), recovered)?;
        Ok(format!("✔ extracted 1 file → {}", file_name(dest)))
    }
}

fn spawn_verify(archive: PathBuf, tx: Sender<JobMsg>) {
    std::thread::spawn(move || {
        let res = do_verify(&archive).map_err(|e| format!("{e:#}"));
        let _ = tx.send(JobMsg::Done(res));
    });
}

fn do_verify(archive: &Path) -> Result<String> {
    let data = std::fs::read(archive)?;
    if data.starts_with(b"CPAS") {
        let files = SolidArchive::unpack(&data)?;
        let total: usize = files.iter().map(|(_, d)| d.len()).sum();
        Ok(format!(
            "✔ verified: {} file(s), {} recovered",
            files.len(),
            human_size(total as u64)
        ))
    } else {
        let recovered = codec::decompress(&data)?;
        Ok(format!("✔ verified: {} recovered", human_size(recovered.len() as u64)))
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn level_label(level: u8) -> String {
    match level {
        1 => "1 — fastest".to_string(),
        5 => "5 — default".to_string(),
        9 => "9 — best ratio".to_string(),
        l => l.to_string(),
    }
}

fn file_name(p: &Path) -> String {
    p.file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| p.to_string_lossy().to_string())
}

fn human_size(n: u64) -> String {
    const KB: f64 = 1024.0;
    let f = n as f64;
    if f >= KB * KB * KB {
        format!("{:.2} GB", f / (KB * KB * KB))
    } else if f >= KB * KB {
        format!("{:.2} MB", f / (KB * KB))
    } else if f >= KB {
        format!("{:.1} KB", f / KB)
    } else {
        format!("{n} B")
    }
}

/// Drop leading separators, `.` and `..` from an archive member name.
fn sanitize_rel(name: &str) -> PathBuf {
    let mut out = PathBuf::new();
    for comp in Path::new(name).components() {
        if let std::path::Component::Normal(c) = comp {
            out.push(c);
        }
    }
    if out.as_os_str().is_empty() {
        out.push("file");
    }
    out
}
