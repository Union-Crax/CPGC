//! Native desktop GUI — a 7-Zip-style file manager built with egui/eframe.
//!
//! Browse folders, tick files/directories, pick a compression level, and
//! **Add** them to a `.cpgc` (single file) or `.cpas` (solid multi-file)
//! archive. **Open** an archive to browse its members and **Extract** chosen
//! ones; **Test** verifies an archive. Long operations run on a background
//! thread and can be **paused, resumed, or cancelled**, with a live progress
//! bar and throughput (MB/s) readout.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::mpsc::Receiver;
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, Result};
use eframe::egui;

use crate::archive::solid::SolidArchive;
use crate::cm::Control;
use crate::codec;

/// Launch the GUI. `start_dir` is the initial directory shown.
pub fn run(start_dir: PathBuf) -> Result<()> {
    let start_dir = start_dir
        .canonicalize()
        .unwrap_or_else(|_| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1000.0, 640.0])
            .with_min_inner_size([620.0, 380.0])
            .with_title("CPGC File Manager"),
        ..Default::default()
    };

    eframe::run_native(
        "CPGC",
        options,
        Box::new(move |cc| {
            cc.egui_ctx.set_visuals(egui::Visuals::light()); // 7-Zip is light
            Box::new(CpgcApp::new(start_dir))
        }),
    )
    .map_err(|e| anyhow!("GUI failed to start: {e}"))
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

struct Entry {
    path: PathBuf,
    name: String,
    is_dir: bool,
    is_archive: bool,
    size: u64,
    modified: Option<SystemTime>,
}

/// State for browsing *inside* an opened archive.
struct ArchiveView {
    path: PathBuf,
    entries: Vec<(String, u64)>, // member name, original size
}

#[derive(PartialEq, Clone, Copy)]
enum StatusKind {
    Info,
    Ok,
    Err,
}

/// Sent from a background worker when the job ends.
enum JobMsg {
    Done(std::result::Result<String, String>),
}

/// A running background operation: cancellable/pausable, with live throughput.
struct Job {
    ctrl: Arc<Control>,
    rx: Receiver<JobMsg>,
    label: String,
    total: u64,
    last_t: Instant,
    last_done: u64,
    speed_mb_s: f64,
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
    archive: Option<ArchiveView>,
    arc_selected: BTreeSet<String>,
    show_add: bool,
    show_about: bool,
    light: bool,
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
            status: "Ready.".to_string(),
            status_kind: StatusKind::Info,
            job: None,
            archive: None,
            arc_selected: BTreeSet::new(),
            show_add: false,
            show_about: false,
            light: true,
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
                let md = e.metadata().ok();
                let is_dir = md.as_ref().map(|m| m.is_dir()).unwrap_or(false);
                let size = md.as_ref().map(|m| m.len()).unwrap_or(0);
                let modified = md.as_ref().and_then(|m| m.modified().ok());
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

    fn open_archive(&mut self, path: PathBuf) {
        let data = match std::fs::read(&path) {
            Ok(d) => d,
            Err(e) => return self.set_status(StatusKind::Err, format!("cannot read archive: {e}")),
        };
        let view = if data.starts_with(b"CPAS") {
            match SolidArchive::list(&data) {
                Ok(entries) => ArchiveView { path: path.clone(), entries },
                Err(e) => return self.set_status(StatusKind::Err, format!("{e:#}")),
            }
        } else if data.starts_with(b"CPGC") {
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
            return self.set_status(StatusKind::Err, "not a CPGC archive");
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

    // --- launching background jobs --------------------------------------

    fn begin_job(&mut self, ctrl: Arc<Control>, rx: Receiver<JobMsg>, label: String, total: u64) {
        self.set_status(StatusKind::Info, label.clone());
        self.job = Some(Job {
            ctrl,
            rx,
            label,
            total,
            last_t: Instant::now(),
            last_done: 0,
            speed_mb_s: 0.0,
        });
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
        let total = total_input_size(&inputs);
        let level = self.level;

        let ctrl = Arc::new(Control::new());
        let (tx, rx) = std::sync::mpsc::channel();
        let label = format!("Compressing {} item(s) → {}", inputs.len(), out);
        {
            let ctrl = ctrl.clone();
            std::thread::spawn(move || {
                let res = do_compress(&inputs, &output, level, &ctrl).map_err(|e| format!("{e:#}"));
                let _ = tx.send(JobMsg::Done(res));
            });
        }
        self.begin_job(ctrl, rx, label, total);
    }

    fn start_extract_members(&mut self, only_selected: bool) {
        if self.busy() {
            return;
        }
        let Some(av) = &self.archive else { return };
        if only_selected && self.arc_selected.is_empty() {
            return;
        }
        let archive = av.path.clone();
        let total: u64 = av.entries.iter().map(|(_, s)| *s).sum();
        let base = strip_ext(&file_name(&archive));
        let parent = archive.parent().unwrap_or(&self.cwd).to_path_buf();
        let dest = if self.extract_dest.trim().is_empty() {
            parent.join(format!("{base}_extracted"))
        } else {
            parent.join(self.extract_dest.trim())
        };
        let wanted = if only_selected { Some(self.arc_selected.clone()) } else { None };

        let ctrl = Arc::new(Control::new());
        let (tx, rx) = std::sync::mpsc::channel();
        let label = format!(
            "Extracting {} from {}",
            if only_selected { format!("{} file(s)", self.arc_selected.len()) } else { "all".into() },
            file_name(&archive)
        );
        {
            let ctrl = ctrl.clone();
            std::thread::spawn(move || {
                let res = do_extract_members(&archive, &dest, wanted.as_ref(), &ctrl)
                    .map_err(|e| format!("{e:#}"));
                let _ = tx.send(JobMsg::Done(res));
            });
        }
        self.begin_job(ctrl, rx, label, total);
    }

    fn start_verify(&mut self, archive: PathBuf) {
        if self.busy() {
            return;
        }
        let total = std::fs::metadata(&archive).map(|m| m.len()).unwrap_or(0);
        let ctrl = Arc::new(Control::new());
        let (tx, rx) = std::sync::mpsc::channel();
        let label = format!("Verifying {}", file_name(&archive));
        {
            let ctrl = ctrl.clone();
            std::thread::spawn(move || {
                let res = do_verify(&archive, &ctrl).map_err(|e| format!("{e:#}"));
                let _ = tx.send(JobMsg::Done(res));
            });
        }
        self.begin_job(ctrl, rx, label, total);
    }

    /// Drain worker messages and update throughput each frame.
    fn poll_job(&mut self) {
        let mut finished: Option<std::result::Result<String, String>> = None;
        if let Some(job) = &mut self.job {
            while let Ok(JobMsg::Done(res)) = job.rx.try_recv() {
                finished = Some(res);
            }
            // Update throughput roughly 4× a second.
            let now = Instant::now();
            let dt = now.duration_since(job.last_t).as_secs_f64();
            if dt >= 0.25 {
                let done = job.ctrl.bytes_done();
                let delta = done.saturating_sub(job.last_done) as f64;
                let inst = delta / dt / 1_000_000.0;
                // Exponential smoothing for a steady readout.
                job.speed_mb_s = if job.speed_mb_s == 0.0 {
                    inst
                } else {
                    job.speed_mb_s * 0.6 + inst * 0.4
                };
                job.last_t = now;
                job.last_done = done;
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
}

// ---------------------------------------------------------------------------
// UI
// ---------------------------------------------------------------------------

impl eframe::App for CpgcApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll_job();
        if self.busy() {
            ctx.request_repaint_after(std::time::Duration::from_millis(100));
        }
        self.menu_bar(ctx);
        self.toolbar(ctx);
        self.status_bar(ctx);
        self.central(ctx);
        self.dialogs(ctx);
    }
}

impl CpgcApp {
    fn menu_bar(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::top("menu").show(ctx, |ui| {
            egui::menu::bar(ui, |ui| {
                ui.menu_button("File", |ui| {
                    if ui.button("Add to archive…").clicked() {
                        self.show_add = !self.selected.is_empty();
                        ui.close_menu();
                    }
                    if ui.button("Exit").clicked() {
                        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                });
                ui.menu_button("View", |ui| {
                    if ui.button("Toggle light/dark").clicked() {
                        self.light = !self.light;
                        ctx.set_visuals(if self.light {
                            egui::Visuals::light()
                        } else {
                            egui::Visuals::dark()
                        });
                        ui.close_menu();
                    }
                    if ui.button("Refresh").clicked() {
                        self.refresh();
                        ui.close_menu();
                    }
                });
                ui.menu_button("Help", |ui| {
                    if ui.button("About CPGC").clicked() {
                        self.show_about = true;
                        ui.close_menu();
                    }
                });
            });
        });
    }

    fn toolbar(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::top("toolbar").show(ctx, |ui| {
            ui.add_space(3.0);
            let busy = self.busy();
            let in_archive = self.archive.is_some();
            ui.horizontal_wrapped(|ui| {
                ui.add_enabled_ui(!busy, |ui| {
                    if ui.button("⬆  Up").clicked() {
                        if in_archive {
                            self.close_archive();
                        } else if let Some(parent) = self.cwd.parent() {
                            self.navigate(parent.to_path_buf());
                        }
                    }
                    ui.separator();
                    if in_archive {
                        if ui
                            .add_enabled(!self.arc_selected.is_empty(), egui::Button::new("➖  Extract"))
                            .clicked()
                        {
                            self.start_extract_members(true);
                        }
                        if ui.button("⏬  Extract all").clicked() {
                            self.start_extract_members(false);
                        }
                        if ui.button("✖  Close").clicked() {
                            self.close_archive();
                        }
                    } else {
                        if ui
                            .add_enabled(!self.selected.is_empty(), egui::Button::new("➕  Add"))
                            .clicked()
                        {
                            self.show_add = true;
                        }
                        if ui.button("ℹ  Info").clicked() {
                            self.info_selected();
                        }
                        ui.separator();
                        ui.label("Level");
                        egui::ComboBox::from_id_source("level")
                            .selected_text(level_label(self.level))
                            .show_ui(ui, |ui| {
                                for lvl in 1u8..=9 {
                                    ui.selectable_value(&mut self.level, lvl, level_label(lvl));
                                }
                            });
                    }
                });
            });
            ui.add_space(2.0);
            ui.horizontal(|ui| {
                let icon = if in_archive { "🗜" } else { "📂" };
                ui.label(icon);
                let path = self
                    .archive
                    .as_ref()
                    .map(|a| a.path.to_string_lossy().to_string())
                    .unwrap_or_else(|| self.cwd.to_string_lossy().to_string());
                ui.monospace(path);
            });
            ui.add_space(3.0);
        });
    }

    fn status_bar(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::bottom("status").show(ctx, |ui| {
            ui.add_space(3.0);
            if let Some(job) = &self.job {
                let done = job.ctrl.bytes_done();
                let frac = if job.total > 0 {
                    (done as f32 / job.total as f32).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                ui.horizontal(|ui| {
                    if job.ctrl.is_paused() {
                        ui.label("⏸");
                    } else {
                        ui.spinner();
                    }
                    ui.label(&job.label);
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.button("✖ Cancel").clicked() {
                            job.ctrl.cancel();
                        }
                        if job.ctrl.is_paused() {
                            if ui.button("▶ Resume").clicked() {
                                job.ctrl.resume();
                            }
                        } else if ui.button("⏸ Pause").clicked() {
                            job.ctrl.pause();
                        }
                        ui.label(format!("{:.2} MB/s", job.speed_mb_s));
                    });
                });
                ui.add(
                    egui::ProgressBar::new(frac)
                        .text(format!("{} / {}", human_size(done), human_size(job.total)))
                        .animate(!job.ctrl.is_paused()),
                );
            } else {
                let n = if self.archive.is_some() {
                    self.arc_selected.len()
                } else {
                    self.selected.len()
                };
                let total = if self.archive.is_some() {
                    self.archive.as_ref().map(|a| a.entries.len()).unwrap_or(0)
                } else {
                    self.entries.len()
                };
                ui.horizontal(|ui| {
                    let color = match self.status_kind {
                        StatusKind::Ok => egui::Color32::from_rgb(0x1d, 0x8a, 0x3f),
                        StatusKind::Err => egui::Color32::from_rgb(0xc0, 0x30, 0x30),
                        StatusKind::Info => ui.visuals().text_color(),
                    };
                    ui.colored_label(color, &self.status);
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(format!("{n} / {total} object(s) selected"));
                    });
                });
            }
            ui.add_space(3.0);
        });
    }

    fn central(&mut self, ctx: &egui::Context) {
        if self.archive.is_some() {
            self.archive_list(ctx);
        } else {
            self.dir_list(ctx);
        }
    }

    fn dir_list(&mut self, ctx: &egui::Context) {
        let mut nav_to: Option<PathBuf> = None;
        let mut toggle: Option<PathBuf> = None;
        let mut open: Option<PathBuf> = None;
        let mut verify: Option<PathBuf> = None;
        let busy = self.busy();

        egui::CentralPanel::default().show(ctx, |ui| {
            egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                egui::Grid::new("files")
                    .num_columns(5)
                    .striped(true)
                    .spacing([14.0, 6.0])
                    .show(ui, |ui| {
                        ui.strong("");
                        ui.strong("Name");
                        ui.strong("Size");
                        ui.strong("Modified");
                        ui.strong("");
                        ui.end_row();

                        for e in &self.entries {
                            let mut sel = self.selected.contains(&e.path);
                            if ui.add_enabled(!busy, egui::Checkbox::without_text(&mut sel)).changed() {
                                toggle = Some(e.path.clone());
                            }
                            let icon = if e.is_dir { "📁" } else if e.is_archive { "🗜" } else { "📄" };
                            let label = format!("{icon} {}", e.name);
                            if e.is_dir {
                                if ui.add_enabled(!busy, egui::Button::new(label).frame(false)).clicked() {
                                    nav_to = Some(e.path.clone());
                                }
                            } else if ui.add(egui::SelectableLabel::new(sel, label)).clicked() {
                                toggle = Some(e.path.clone());
                            }
                            ui.monospace(if e.is_dir { String::new() } else { human_size(e.size) });
                            ui.monospace(e.modified.map(fmt_mtime).unwrap_or_default());
                            ui.horizontal(|ui| {
                                if e.is_archive {
                                    ui.add_enabled_ui(!busy, |ui| {
                                        if ui.button("Open").clicked() {
                                            open = Some(e.path.clone());
                                        }
                                        if ui.button("Test").clicked() {
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
        if let Some(p) = open {
            self.open_archive(p);
        }
        if let Some(p) = verify {
            self.start_verify(p);
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
                    .spacing([14.0, 6.0])
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
                                if ui
                                    .add(egui::SelectableLabel::new(sel, format!("📄 {name}")))
                                    .clicked()
                                {
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

    fn dialogs(&mut self, ctx: &egui::Context) {
        if self.show_add {
            let mut open = true;
            let mut go = false;
            egui::Window::new("Add to Archive")
                .collapsible(false)
                .resizable(false)
                .open(&mut open)
                .show(ctx, |ui| {
                    egui::Grid::new("add").num_columns(2).spacing([10.0, 8.0]).show(ui, |ui| {
                        ui.label("Archive name");
                        ui.add(egui::TextEdit::singleline(&mut self.out_name).desired_width(220.0));
                        ui.end_row();
                        ui.label("Compression level");
                        egui::ComboBox::from_id_source("add_level")
                            .selected_text(level_label(self.level))
                            .show_ui(ui, |ui| {
                                for lvl in 1u8..=9 {
                                    ui.selectable_value(&mut self.level, lvl, level_label(lvl));
                                }
                            });
                        ui.end_row();
                        ui.label("Items");
                        ui.label(format!("{} selected", self.selected.len()));
                        ui.end_row();
                    });
                    ui.separator();
                    ui.horizontal(|ui| {
                        if ui.button("OK").clicked() {
                            go = true;
                        }
                        if ui.button("Cancel").clicked() {
                            self.show_add = false;
                        }
                    });
                });
            if go {
                self.show_add = false;
                self.start_compress();
            } else if !open {
                self.show_add = false;
            }
        }

        if self.show_about {
            let mut open = true;
            egui::Window::new("About CPGC")
                .collapsible(false)
                .resizable(false)
                .open(&mut open)
                .show(ctx, |ui| {
                    ui.heading("CPGC File Manager");
                    ui.label("Context-mixing compressor with its own CPGC-NX engine.");
                    ui.label("Archives: .cpgc (single file) and .cpas (solid multi-file).");
                });
            self.show_about = open;
        }
    }

    fn info_selected(&mut self) {
        let archive = self.selected.iter().find(|p| {
            let n = p.to_string_lossy().to_lowercase();
            n.ends_with(".cpgc") || n.ends_with(".cpas")
        });
        let Some(path) = archive.cloned() else {
            return self.set_status(StatusKind::Info, "Select a .cpgc/.cpas archive for info.");
        };
        match std::fs::read(&path) {
            Ok(data) if data.starts_with(b"CPAS") => match SolidArchive::list(&data) {
                Ok(list) => self.set_status(
                    StatusKind::Info,
                    format!("{}: {} file(s), {} packed", file_name(&path), list.len(), human_size(data.len() as u64)),
                ),
                Err(e) => self.set_status(StatusKind::Err, format!("{e:#}")),
            },
            Ok(data) if data.starts_with(b"CPGC") => {
                let orig = if data.len() >= 14 {
                    u64::from_le_bytes(data[6..14].try_into().unwrap())
                } else {
                    0
                };
                self.set_status(
                    StatusKind::Info,
                    format!(
                        "{}: {} → {} (ratio {:.3})",
                        file_name(&path),
                        human_size(orig),
                        human_size(data.len() as u64),
                        data.len() as f64 / orig.max(1) as f64
                    ),
                );
            }
            _ => self.set_status(StatusKind::Err, "not a CPGC archive"),
        }
    }
}

// ---------------------------------------------------------------------------
// Background work (runs off the UI thread)
// ---------------------------------------------------------------------------

fn do_compress(inputs: &[PathBuf], output: &Path, level: u8, ctrl: &Control) -> Result<String> {
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

    let packed = if files.len() == 1 {
        codec::compress_with_control(&files[0].1, level, ctrl)?
    } else {
        let pairs: Vec<(&str, &[u8])> =
            files.iter().map(|(n, d)| (n.as_str(), d.as_slice())).collect();
        SolidArchive::pack_with_control(&pairs, level, ctrl)?
    };
    std::fs::write(output, &packed)?;

    let pct = packed.len() as f64 / total_raw.max(1) as f64 * 100.0;
    Ok(format!(
        "Done — {} → {} ({:.1}% of original, {} file(s))",
        human_size(total_raw as u64),
        human_size(packed.len() as u64),
        pct,
        files.len()
    ))
}

fn do_extract_members(
    archive: &Path,
    dest: &Path,
    wanted: Option<&BTreeSet<String>>,
    ctrl: &Control,
) -> Result<String> {
    let data = std::fs::read(archive)?;
    std::fs::create_dir_all(dest)?;
    let want = |name: &str| wanted.map(|w| w.contains(name)).unwrap_or(true);

    if data.starts_with(b"CPAS") {
        let files = SolidArchive::unpack_with_control(&data, ctrl)?;
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
        Ok(format!("Done — extracted {} file(s) → {}", count, file_name(dest)))
    } else {
        let recovered = codec::decompress_with_control(&data, ctrl)?;
        let stem = archive
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "recovered".to_string());
        if !want(&stem) {
            return Ok("nothing selected".to_string());
        }
        std::fs::write(dest.join(&stem), recovered)?;
        Ok(format!("Done — extracted 1 file → {}", file_name(dest)))
    }
}

fn do_verify(archive: &Path, ctrl: &Control) -> Result<String> {
    let data = std::fs::read(archive)?;
    if data.starts_with(b"CPAS") {
        let files = SolidArchive::unpack_with_control(&data, ctrl)?;
        let total: usize = files.iter().map(|(_, d)| d.len()).sum();
        Ok(format!("OK — {} file(s), {} recovered", files.len(), human_size(total as u64)))
    } else {
        let recovered = codec::decompress_with_control(&data, ctrl)?;
        Ok(format!("OK — {} recovered", human_size(recovered.len() as u64)))
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn total_input_size(inputs: &[PathBuf]) -> u64 {
    let mut total = 0u64;
    for p in inputs {
        if p.is_dir() {
            for e in walkdir::WalkDir::new(p).into_iter().flatten() {
                if e.file_type().is_file() {
                    total += e.metadata().map(|m| m.len()).unwrap_or(0);
                }
            }
        } else {
            total += std::fs::metadata(p).map(|m| m.len()).unwrap_or(0);
        }
    }
    total
}

fn level_label(level: u8) -> String {
    match level {
        1 => "1 — fastest".to_string(),
        5 => "5 — normal".to_string(),
        9 => "9 — ultra".to_string(),
        l => l.to_string(),
    }
}

fn file_name(p: &Path) -> String {
    p.file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| p.to_string_lossy().to_string())
}

fn strip_ext(name: &str) -> String {
    let lower = name.to_lowercase();
    if lower.ends_with(".cpgc") || lower.ends_with(".cpas") {
        name[..name.len() - 5].to_string()
    } else {
        name.to_string()
    }
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

/// Format a modification time as `YYYY-MM-DD HH:MM` (UTC, no external deps).
fn fmt_mtime(m: SystemTime) -> String {
    let secs = m
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let days = secs.div_euclid(86400);
    let tod = secs.rem_euclid(86400);
    let (y, mo, d) = civil_from_days(days);
    format!("{:04}-{:02}-{:02} {:02}:{:02}", y, mo, d, tod / 3600, (tod % 3600) / 60)
}

/// Howard Hinnant's days-from-epoch → civil date algorithm.
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719468;
    let era = (if z >= 0 { z } else { z - 146096 }) / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = (if mp < 10 { mp + 3 } else { mp - 9 }) as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
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
        let files: Vec<(&str, &[u8])> = vec![
            ("a.txt", b"alpha alpha alpha"),
            ("sub/b.bin", &[1u8, 2, 3, 4, 5, 6]),
            ("c.log", b"gamma gamma gamma gamma"),
        ];
        let packed = SolidArchive::pack(&files, 5).unwrap();
        let arc = dir.path().join("test.cpas");
        std::fs::write(&arc, &packed).unwrap();

        let mut wanted = BTreeSet::new();
        wanted.insert("a.txt".to_string());
        wanted.insert("sub/b.bin".to_string());
        let dest = dir.path().join("out");
        do_extract_members(&arc, &dest, Some(&wanted), &Control::new()).unwrap();

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
        do_extract_members(&arc, &dest, None, &Control::new()).unwrap();
        assert_eq!(std::fs::read(dest.join("x")).unwrap(), b"one");
        assert_eq!(std::fs::read(dest.join("y")).unwrap(), b"two");
    }

    #[test]
    fn civil_date_known_values() {
        // 2021-01-01 was day 18628 since the Unix epoch.
        assert_eq!(civil_from_days(18628), (2021, 1, 1));
        assert_eq!(civil_from_days(0), (1970, 1, 1));
    }
}
