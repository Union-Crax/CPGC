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

use anyhow::{anyhow, Result};
use eframe::egui;

use crate::archive::solid::SolidArchive;
use crate::codec;

/// Launch the GUI. `start_dir` is the initial directory shown; if `open_archive`
/// is set (Explorer "Open with CPGC"), that archive is opened on start.
pub fn run(start_dir: PathBuf, open_archive: Option<PathBuf>) -> Result<()> {
    let start_dir = start_dir
        .canonicalize()
        .unwrap_or_else(|_| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
    let open_archive = open_archive.and_then(|p| p.canonicalize().ok());

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
        Box::new(move |_cc| Box::new(CpgcApp::new(start_dir, open_archive))),
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
    modified: Option<std::time::SystemTime>,
}

impl Entry {
    /// 7-Zip-style "Type" column text.
    fn type_label(&self) -> &'static str {
        if self.is_dir {
            "Folder"
        } else if self.is_archive {
            "CPGC Archive"
        } else {
            "File"
        }
    }
}

/// State for browsing *inside* an opened archive (7-Zip style).
struct ArchiveView {
    path: PathBuf,
    entries: Vec<(String, u64)>, // member name, original size
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
    // When Some, the file list shows the contents of this archive instead of
    // the directory, and `arc_selected` tracks ticked member names.
    archive: Option<ArchiveView>,
    arc_selected: BTreeSet<String>,
}

impl CpgcApp {
    fn new(cwd: PathBuf, open_archive: Option<PathBuf>) -> Self {
        let mut app = Self {
            cwd,
            entries: Vec::new(),
            selected: BTreeSet::new(),
            level: 5,
            out_name: "archive.cpgc".to_string(),
            extract_dest: String::new(),
            status: "Ready. Tick items to compress, or open an archive to browse it.".to_string(),
            status_kind: StatusKind::Info,
            job: None,
            archive: None,
            arc_selected: BTreeSet::new(),
        };
        app.refresh();
        // Launched via the Explorer "Open with CPGC" verb: jump straight into
        // browsing the archive.
        if let Some(arc) = open_archive {
            app.open_archive(arc);
        }
        app
    }

    fn refresh(&mut self) {
        self.entries.clear();
        if let Ok(rd) = std::fs::read_dir(&self.cwd) {
            for e in rd.flatten() {
                let path = e.path();
                let name = e.file_name().to_string_lossy().to_string();
                let (is_dir, size, modified) = match e.metadata() {
                    Ok(m) => (m.is_dir(), m.len(), m.modified().ok()),
                    Err(_) => (false, 0, None),
                };
                let lower = name.to_lowercase();
                let is_archive = lower.ends_with(".cpgc") || lower.ends_with(".cpas");
                self.entries.push(Entry { path, name, is_dir, is_archive, size, modified });
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

    /// Open an archive and show its members (does not decompress for solid
    /// archives — the table is read directly).
    fn open_archive(&mut self, path: PathBuf) {
        let data = match std::fs::read(&path) {
            Ok(d) => d,
            Err(e) => {
                self.set_status(StatusKind::Err, format!("cannot read archive: {e}"));
                return;
            }
        };
        let view = if data.starts_with(b"CPAS") {
            match SolidArchive::list(&data) {
                Ok(entries) => ArchiveView { path: path.clone(), entries },
                Err(e) => {
                    self.set_status(StatusKind::Err, format!("{e:#}"));
                    return;
                }
            }
        } else if data.starts_with(b"CPGC") {
            // Single-file archive: present the one original file it holds.
            let orig = if data.len() >= 14 {
                u64::from_le_bytes(data[6..14].try_into().unwrap())
            } else {
                0
            };
            let name = path
                .file_stem()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| "file".to_string());
            ArchiveView { path: path.clone(), entries: vec![(name, orig)] }
        } else {
            self.set_status(StatusKind::Err, "not a CPGC archive");
            return;
        };
        let n = view.entries.len();
        self.archive = Some(view);
        self.arc_selected.clear();
        self.extract_dest.clear();
        self.set_status(StatusKind::Info, format!("Opened {} ({} file(s))", file_name(&path), n));
    }

    fn close_archive(&mut self) {
        self.archive = None;
        self.arc_selected.clear();
        self.refresh();
    }

    /// Extract members of the open archive. `only_selected` limits to ticked
    /// members; otherwise everything is extracted.
    fn start_extract_members(&mut self, only_selected: bool) {
        if self.busy() {
            return;
        }
        let Some(av) = &self.archive else { return };
        if only_selected && self.arc_selected.is_empty() {
            return;
        }
        let archive = av.path.clone();
        let base = archive
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "extracted".to_string());
        let parent = archive.parent().unwrap_or(&self.cwd).to_path_buf();
        let dest = if self.extract_dest.trim().is_empty() {
            parent.join(format!("{base}_extracted"))
        } else {
            parent.join(self.extract_dest.trim())
        };
        let wanted = if only_selected {
            Some(self.arc_selected.clone())
        } else {
            None
        };

        let (tx, rx) = std::sync::mpsc::channel();
        let label = format!(
            "Extracting {} from {}",
            if only_selected { format!("{} file(s)", self.arc_selected.len()) } else { "all".into() },
            file_name(&archive)
        );
        self.set_status(StatusKind::Info, label.clone());
        spawn_extract_members(archive, dest, wanted, tx);
        self.job = Some(Job { rx, label, progress: 0.0 });
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
        self.job = Some(Job { rx, label, progress: 0.0 });
    }

    fn start_verify(&mut self, archive: PathBuf) {
        if self.busy() {
            return;
        }
        let (tx, rx) = std::sync::mpsc::channel();
        let label = format!("Verifying {}", file_name(&archive));
        self.set_status(StatusKind::Info, label.clone());
        spawn_verify(archive, tx);
        self.job = Some(Job { rx, label, progress: 0.0 });
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
            if self.archive.is_some() {
                self.archive_toolbar(ui);
            } else {
                self.files_toolbar(ui);
            }
            ui.add_space(4.0);
        });
    }

    fn files_toolbar(&mut self, ui: &mut egui::Ui) {
        let busy = self.busy();
        let mut nav_to: Option<PathBuf> = None;
        ui.horizontal_wrapped(|ui| {
            ui.heading("📦 CPGC");
            ui.separator();
            ui.add_enabled_ui(!busy, |ui| {
                if ui.button("⬆ Up").on_hover_text("Parent folder").clicked() {
                    if let Some(parent) = self.cwd.parent() {
                        nav_to = Some(parent.to_path_buf());
                    }
                }
                if ui.button("🔄").on_hover_text("Refresh").clicked() {
                    self.refresh();
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
                if ui
                    .button("🗜 Add to archive")
                    .on_hover_text("Compress the ticked items")
                    .clicked()
                {
                    self.start_compress();
                }
            });
        });
        ui.add_space(2.0);
        // Clickable breadcrumb address bar (each ancestor navigates there).
        ui.horizontal_wrapped(|ui| {
            ui.label("📂");
            let mut ancestors: Vec<PathBuf> = self.cwd.ancestors().map(|p| p.to_path_buf()).collect();
            ancestors.reverse();
            let last = ancestors.len().saturating_sub(1);
            for (i, p) in ancestors.iter().enumerate() {
                let label = p
                    .file_name()
                    .map(|s| s.to_string_lossy().to_string())
                    .unwrap_or_else(|| p.to_string_lossy().to_string());
                if i == last {
                    ui.strong(label);
                } else {
                    if ui.add(egui::Button::new(label).frame(false)).clicked() {
                        nav_to = Some(p.clone());
                    }
                    ui.weak("›");
                }
            }
        });
        if let Some(p) = nav_to {
            self.navigate(p);
        }
    }

    fn archive_toolbar(&mut self, ui: &mut egui::Ui) {
        let busy = self.busy();
        let (name, n) = self
            .archive
            .as_ref()
            .map(|a| (file_name(&a.path), a.entries.len()))
            .unwrap_or_default();
        ui.horizontal_wrapped(|ui| {
            ui.heading("🗜 Archive");
            ui.separator();
            ui.add_enabled_ui(!busy, |ui| {
                if ui.button("✖ Close").clicked() {
                    self.close_archive();
                }
            });
            ui.label(format!("{} selected of {}", self.arc_selected.len(), n));
            ui.separator();
            ui.label("Extract to");
            ui.add(
                egui::TextEdit::singleline(&mut self.extract_dest)
                    .hint_text(format!("{}_extracted", strip_ext(&name)))
                    .desired_width(180.0),
            );
            ui.add_enabled_ui(!busy && !self.arc_selected.is_empty(), |ui| {
                if ui.button("⬇ Extract selected").clicked() {
                    self.start_extract_members(true);
                }
            });
            ui.add_enabled_ui(!busy, |ui| {
                if ui.button("⬇ Extract all").clicked() {
                    self.start_extract_members(false);
                }
            });
            ui.separator();
            let arc_path = self.archive.as_ref().map(|a| a.path.clone());
            ui.add_enabled_ui(!busy, |ui| {
                if ui.button("🧪 Test").on_hover_text("Verify the archive decodes correctly").clicked() {
                    if let Some(p) = arc_path {
                        self.start_verify(p);
                    }
                }
            });
        });
        ui.add_space(2.0);
        ui.horizontal(|ui| {
            ui.label("🗜");
            ui.monospace(&name);
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
        if self.archive.is_some() {
            self.archive_list(ctx);
        } else {
            self.dir_list(ctx);
        }
    }

    fn dir_list(&mut self, ctx: &egui::Context) {
        // Actions deferred until after the immutable iteration over entries.
        let mut nav_to: Option<PathBuf> = None;
        let mut toggle: Option<PathBuf> = None;
        let mut open: Option<PathBuf> = None;
        let mut select_all: Option<bool> = None;
        let busy = self.busy();
        let all_selected = !self.entries.is_empty()
            && self.entries.iter().all(|e| self.selected.contains(&e.path));

        egui::CentralPanel::default().show(ctx, |ui| {
            egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                egui::Grid::new("files")
                    .num_columns(5)
                    .striped(true)
                    .spacing([14.0, 6.0])
                    .show(ui, |ui| {
                        // Header row: the first cell is a select-all checkbox.
                        let mut all = all_selected;
                        if ui.add_enabled(!busy, egui::Checkbox::without_text(&mut all))
                            .on_hover_text("Select all / none")
                            .changed()
                        {
                            select_all = Some(all);
                        }
                        ui.strong("Name");
                        ui.strong("Size");
                        ui.strong("Modified");
                        ui.strong("Type");
                        ui.end_row();

                        for e in &self.entries {
                            // Selection checkbox (files and folders both selectable).
                            let mut sel = self.selected.contains(&e.path);
                            if ui.add_enabled(!busy, egui::Checkbox::without_text(&mut sel)).changed() {
                                toggle = Some(e.path.clone());
                            }

                            // Name — single click selects, double click opens
                            // (navigate into folders, browse into archives).
                            let icon = if e.is_dir {
                                "📁"
                            } else if e.is_archive {
                                "🗜"
                            } else {
                                "📄"
                            };
                            let label = format!("{icon} {}", e.name);
                            let resp = ui
                                .add(egui::SelectableLabel::new(sel, label))
                                .on_hover_text(if e.is_dir || e.is_archive {
                                    "Double-click to open"
                                } else {
                                    "Click to select"
                                });
                            if resp.double_clicked() {
                                if e.is_dir {
                                    nav_to = Some(e.path.clone());
                                } else if e.is_archive {
                                    open = Some(e.path.clone());
                                }
                            } else if resp.clicked() {
                                toggle = Some(e.path.clone());
                            }

                            // Size.
                            if e.is_dir {
                                ui.label("—");
                            } else {
                                ui.monospace(human_size(e.size));
                            }

                            // Modified + Type columns (7-Zip style).
                            ui.label(format_modified(e.modified));
                            ui.label(e.type_label());
                            ui.end_row();
                        }
                    });
            });
        });

        if let Some(all) = select_all {
            if all {
                for e in &self.entries {
                    self.selected.insert(e.path.clone());
                }
            } else {
                self.selected.clear();
            }
        }
        if let Some(p) = toggle {
            if !self.selected.remove(&p) {
                self.selected.insert(p);
            }
        }
        if let Some(p) = nav_to {
            self.navigate(p);
        }
        if let Some(p) = open {
            self.open_archive(p);
        }
    }

    fn archive_list(&mut self, ctx: &egui::Context) {
        let mut toggle: Option<String> = None;
        let busy = self.busy();

        egui::CentralPanel::default().show(ctx, |ui| {
            egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                egui::Grid::new("members")
                    .num_columns(3)
                    .striped(true)
                    .spacing([12.0, 6.0])
                    .show(ui, |ui| {
                        ui.strong("");
                        ui.strong("Name");
                        ui.strong("Size");
                        ui.end_row();

                        if let Some(av) = &self.archive {
                            for (name, size) in &av.entries {
                                let mut sel = self.arc_selected.contains(name);
                                if ui
                                    .add_enabled(!busy, egui::Checkbox::without_text(&mut sel))
                                    .changed()
                                {
                                    toggle = Some(name.clone());
                                }
                                let label = format!("📄 {name}");
                                if ui.add(egui::SelectableLabel::new(sel, label)).clicked() {
                                    toggle = Some(name.clone());
                                }
                                ui.monospace(human_size(*size));
                                ui.end_row();
                            }
                        }
                    });
            });
        });

        if let Some(name) = toggle {
            if !self.arc_selected.remove(&name) {
                self.arc_selected.insert(name);
            }
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

fn spawn_extract_members(
    archive: PathBuf,
    dest: PathBuf,
    wanted: Option<BTreeSet<String>>,
    tx: Sender<JobMsg>,
) {
    std::thread::spawn(move || {
        let res = do_extract_members(&archive, &dest, wanted.as_ref()).map_err(|e| format!("{e:#}"));
        let _ = tx.send(JobMsg::Done(res));
    });
}

fn do_extract_members(
    archive: &Path,
    dest: &Path,
    wanted: Option<&BTreeSet<String>>,
) -> Result<String> {
    let data = std::fs::read(archive)?;
    std::fs::create_dir_all(dest)?;
    let want = |name: &str| wanted.map(|w| w.contains(name)).unwrap_or(true);

    if data.starts_with(b"CPAS") {
        // Solid: the whole stream must be decompressed, then chosen members written.
        let files = SolidArchive::unpack(&data)?;
        let mut count = 0usize;
        for (name, bytes) in &files {
            if !want(name) {
                continue;
            }
            let out = dest.join(sanitize_rel(name));
            if let Some(parent) = out.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&out, bytes)?;
            count += 1;
        }
        Ok(format!("✔ extracted {} file(s) → {}", count, file_name(dest)))
    } else {
        let recovered = codec::decompress(&data)?;
        let stem = archive
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "recovered".to_string());
        if !want(&stem) {
            return Ok("nothing selected".to_string());
        }
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

/// Strip a trailing `.cpgc` / `.cpas` extension from a file name.
fn strip_ext(name: &str) -> String {
    let lower = name.to_lowercase();
    if lower.ends_with(".cpgc") || lower.ends_with(".cpas") {
        name[..name.len() - 5].to_string()
    } else {
        name.to_string()
    }
}

/// Format a file's modified time as `YYYY-MM-DD HH:MM` (UTC). Kept dependency
/// free — the project has no date/time crate — so this shows UTC rather than
/// local time.
fn format_modified(t: Option<std::time::SystemTime>) -> String {
    let Some(t) = t else { return "—".to_string() };
    let secs = match t.duration_since(std::time::UNIX_EPOCH) {
        Ok(d) => d.as_secs() as i64,
        Err(_) => return "—".to_string(),
    };
    let days = secs.div_euclid(86_400);
    let rem = secs.rem_euclid(86_400);
    let (h, mi) = (rem / 3600, (rem % 3600) / 60);
    let (y, m, d) = civil_from_days(days);
    format!("{y:04}-{m:02}-{d:02} {h:02}:{mi:02}")
}

/// Convert days-since-1970-01-01 into a (year, month, day) civil date.
/// Howard Hinnant's `civil_from_days` algorithm.
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32; // [1, 31]
    let m = (if mp < 10 { mp + 3 } else { mp - 9 }) as u32; // [1, 12]
    (if m <= 2 { y + 1 } else { y }, m, d)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_selected_members_only() {
        let dir = tempfile::tempdir().unwrap();
        // Build a solid archive with three members.
        let files: Vec<(&str, &[u8])> = vec![
            ("a.txt", b"alpha alpha alpha"),
            ("sub/b.bin", &[1u8, 2, 3, 4, 5, 6]),
            ("c.log", b"gamma gamma gamma gamma"),
        ];
        let packed = SolidArchive::pack(&files, 5).unwrap();
        let arc = dir.path().join("test.cpas");
        std::fs::write(&arc, &packed).unwrap();

        // Extract only the two we want; the third must NOT appear.
        let mut wanted = BTreeSet::new();
        wanted.insert("a.txt".to_string());
        wanted.insert("sub/b.bin".to_string());
        let dest = dir.path().join("out");
        do_extract_members(&arc, &dest, Some(&wanted)).unwrap();

        assert_eq!(std::fs::read(dest.join("a.txt")).unwrap(), b"alpha alpha alpha");
        assert_eq!(std::fs::read(dest.join("sub/b.bin")).unwrap(), vec![1, 2, 3, 4, 5, 6]);
        assert!(!dest.join("c.log").exists(), "unselected member was extracted");
    }

    #[test]
    fn extract_all_members() {
        let dir = tempfile::tempdir().unwrap();
        let files: Vec<(&str, &[u8])> = vec![("x", b"one"), ("y", b"two")];
        let packed = SolidArchive::pack(&files, 5).unwrap();
        let arc = dir.path().join("all.cpas");
        std::fs::write(&arc, &packed).unwrap();

        let dest = dir.path().join("all_out");
        do_extract_members(&arc, &dest, None).unwrap();
        assert_eq!(std::fs::read(dest.join("x")).unwrap(), b"one");
        assert_eq!(std::fs::read(dest.join("y")).unwrap(), b"two");
    }
}
