//! Built-in local web GUI — a 7-Zip-style file manager served over HTTP.
//!
//! `cpgc gui` starts a small server bound to localhost and prints a URL. Open
//! it in any browser to browse files, compress selections into `.cpgc`
//! archives, inspect archives, and extract them — no GL/X11/display required,
//! so it works the same on a headless server and a desktop.
//!
//! All file access happens server-side and is sandboxed under a `--root`
//! directory (default: the working directory). The HTTP API is intentionally
//! tiny: form-encoded requests, hand-built JSON responses, no async runtime.

use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use tiny_http::{Header, Method, Response, Server};

use crate::archive::solid::SolidArchive;
use crate::codec;

/// Start the GUI server and serve requests until the process is killed.
pub fn run(port: u16, root: PathBuf) -> Result<()> {
    let root = root
        .canonicalize()
        .with_context(|| format!("resolving root {:?}", root))?;
    let addr = format!("127.0.0.1:{port}");
    let server = Server::http(&addr).map_err(|e| anyhow!("starting server: {e}"))?;

    println!("┌──────────────────────────────────────────────┐");
    println!("│  CPGC GUI is running                           │");
    println!("│  → open  http://{addr}            ", );
    println!("│  root: {}", root.display());
    println!("│  press Ctrl-C to stop                          │");
    println!("└──────────────────────────────────────────────┘");

    for request in server.incoming_requests() {
        if let Err(e) = handle(request, &root) {
            eprintln!("request error: {e:#}");
        }
    }
    Ok(())
}

fn handle(mut request: tiny_http::Request, root: &Path) -> Result<()> {
    let url = request.url().to_string();
    let method = request.method().clone();
    let (path, query) = split_query(&url);

    // Read the request body (small form-encoded payloads only).
    let mut body = String::new();
    if matches!(method, Method::Post) {
        request.as_reader().read_to_string(&mut body).ok();
    }

    let result = match (&method, path) {
        (Method::Get, "/") => respond_html(request, INDEX_HTML),
        (Method::Get, "/api/ls") => {
            let json = api_ls(root, &query);
            respond_json(request, &json)
        }
        (Method::Get, "/api/info") => {
            let json = api_info(root, &query);
            respond_json(request, &json)
        }
        (Method::Post, "/api/compress") => {
            let json = api_compress(root, &body);
            respond_json(request, &json)
        }
        (Method::Post, "/api/extract") => {
            let json = api_extract(root, &body);
            respond_json(request, &json)
        }
        _ => request
            .respond(Response::from_string("not found").with_status_code(404))
            .map_err(|e| anyhow!("{e}")),
    };
    result
}

// ---------------------------------------------------------------------------
// API handlers — each returns a JSON string.
// ---------------------------------------------------------------------------

fn api_ls(root: &Path, query: &str) -> String {
    let p = query_param(query, "path").unwrap_or_default();
    let dir = match resolve_existing(root, &p) {
        Some(d) if d.is_dir() => d,
        _ => root.to_path_buf(),
    };

    let mut entries: Vec<(String, bool, u64)> = Vec::new();
    if let Ok(rd) = std::fs::read_dir(&dir) {
        for e in rd.flatten() {
            let name = e.file_name().to_string_lossy().to_string();
            let (is_dir, size) = match e.metadata() {
                Ok(m) => (m.is_dir(), m.len()),
                Err(_) => (false, 0),
            };
            entries.push((name, is_dir, size));
        }
    }
    // Directories first, then files, each alphabetical (case-insensitive).
    entries.sort_by(|a, b| {
        b.1.cmp(&a.1)
            .then(a.0.to_lowercase().cmp(&b.0.to_lowercase()))
    });

    let parent = dir
        .parent()
        .filter(|pp| pp.starts_with(root) || *pp == root)
        .map(|pp| pp.to_string_lossy().to_string())
        .filter(|_| dir != root);

    let mut items = String::new();
    for (i, (name, is_dir, size)) in entries.iter().enumerate() {
        if i > 0 {
            items.push(',');
        }
        let lower = name.to_lowercase();
        let is_archive = lower.ends_with(".cpgc") || lower.ends_with(".cpas");
        items.push_str(&format!(
            "{{\"name\":{},\"dir\":{},\"size\":{},\"archive\":{}}}",
            json_str(name),
            is_dir,
            size,
            is_archive
        ));
    }
    format!(
        "{{\"path\":{},\"parent\":{},\"entries\":[{}]}}",
        json_str(&dir.to_string_lossy()),
        match parent {
            Some(p) => json_str(&p),
            None => "null".to_string(),
        },
        items
    )
}

fn api_info(root: &Path, query: &str) -> String {
    let p = query_param(query, "path").unwrap_or_default();
    let file = match resolve_existing(root, &p) {
        Some(f) if f.is_file() => f,
        _ => return err_json("not a file"),
    };
    let data = match std::fs::read(&file) {
        Ok(d) => d,
        Err(e) => return err_json(&format!("read failed: {e}")),
    };
    if data.starts_with(b"CPAS") {
        match SolidArchive::list(&data) {
            Ok(list) => {
                let mut items = String::new();
                for (i, (name, size)) in list.iter().enumerate() {
                    if i > 0 {
                        items.push(',');
                    }
                    items.push_str(&format!(
                        "{{\"name\":{},\"size\":{}}}",
                        json_str(name),
                        size
                    ));
                }
                format!(
                    "{{\"ok\":true,\"kind\":\"solid\",\"compressed\":{},\"files\":[{}]}}",
                    data.len(),
                    items
                )
            }
            Err(e) => err_json(&format!("{e}")),
        }
    } else if data.starts_with(b"CPGC") {
        let orig = if data.len() >= 14 {
            u64::from_le_bytes(data[6..14].try_into().unwrap())
        } else {
            0
        };
        format!(
            "{{\"ok\":true,\"kind\":\"single\",\"compressed\":{},\"original\":{},\"ratio\":{:.4}}}",
            data.len(),
            orig,
            data.len() as f64 / orig.max(1) as f64
        )
    } else {
        err_json("not a CPGC archive")
    }
}

fn api_compress(root: &Path, body: &str) -> String {
    let inputs: Vec<String> = query_param(body, "inputs")
        .unwrap_or_default()
        .split('\n')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    let output = query_param(body, "output").unwrap_or_default();
    let level: u8 = query_param(body, "level")
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);

    if inputs.is_empty() || output.is_empty() {
        return err_json("select at least one item and an output name");
    }
    let out_path = match resolve_new(root, &output) {
        Some(p) => p,
        None => return err_json("output path is outside the allowed root"),
    };

    match do_compress(root, &inputs, &out_path, level) {
        Ok((orig, comp, nfiles)) => format!(
            "{{\"ok\":true,\"output\":{},\"original\":{},\"compressed\":{},\"ratio\":{:.4},\"files\":{}}}",
            json_str(&out_path.to_string_lossy()),
            orig,
            comp,
            comp as f64 / orig.max(1) as f64,
            nfiles
        ),
        Err(e) => err_json(&format!("{e:#}")),
    }
}

fn do_compress(
    root: &Path,
    inputs: &[String],
    out_path: &Path,
    level: u8,
) -> Result<(usize, usize, usize)> {
    // Resolve and gather (relative_name, bytes) for every input file.
    let mut files: Vec<(String, Vec<u8>)> = Vec::new();
    for inp in inputs {
        let p = resolve_existing(root, inp).ok_or_else(|| anyhow!("not found: {inp}"))?;
        if p.is_dir() {
            for entry in walkdir::WalkDir::new(&p).sort_by_file_name() {
                let entry = entry?;
                if !entry.file_type().is_file() {
                    continue;
                }
                let rel = entry
                    .path()
                    .strip_prefix(p.parent().unwrap_or(&p))
                    .unwrap_or(entry.path())
                    .to_string_lossy()
                    .replace('\\', "/");
                files.push((rel, std::fs::read(entry.path())?));
            }
        } else {
            let name = p
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_else(|| "file".to_string());
            files.push((name, std::fs::read(&p)?));
        }
    }
    if files.is_empty() {
        return Err(anyhow!("nothing to compress"));
    }
    let total_raw: usize = files.iter().map(|(_, d)| d.len()).sum();

    // One plain file → single-file .cpgc. Otherwise a solid CPAS archive.
    let packed = if files.len() == 1 {
        codec::compress(&files[0].1, level)?
    } else {
        let pairs: Vec<(&str, &[u8])> =
            files.iter().map(|(n, d)| (n.as_str(), d.as_slice())).collect();
        SolidArchive::pack(&pairs, level)?
    };
    std::fs::write(out_path, &packed)
        .with_context(|| format!("writing {:?}", out_path))?;
    Ok((total_raw, packed.len(), files.len()))
}

fn api_extract(root: &Path, body: &str) -> String {
    let archive = query_param(body, "archive").unwrap_or_default();
    let dest = query_param(body, "dest").unwrap_or_default();
    if archive.is_empty() || dest.is_empty() {
        return err_json("archive and destination are required");
    }
    let arc_path = match resolve_existing(root, &archive) {
        Some(p) if p.is_file() => p,
        _ => return err_json("archive not found"),
    };
    let dest_path = match resolve_new(root, &dest) {
        Some(p) => p,
        None => return err_json("destination is outside the allowed root"),
    };

    match do_extract(&arc_path, &dest_path) {
        Ok(n) => format!(
            "{{\"ok\":true,\"dest\":{},\"files\":{}}}",
            json_str(&dest_path.to_string_lossy()),
            n
        ),
        Err(e) => err_json(&format!("{e:#}")),
    }
}

fn do_extract(arc_path: &Path, dest: &Path) -> Result<usize> {
    let data = std::fs::read(arc_path)?;
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
        Ok(files.len())
    } else {
        let recovered = codec::decompress(&data)?;
        // Strip a trailing .cpgc to name the recovered file.
        let stem = arc_path
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "recovered".to_string());
        std::fs::write(dest.join(stem), recovered)?;
        Ok(1)
    }
}

// ---------------------------------------------------------------------------
// Path safety
// ---------------------------------------------------------------------------

/// Resolve an existing path and ensure it stays under `root`.
fn resolve_existing(root: &Path, p: &str) -> Option<PathBuf> {
    let cand = PathBuf::from(p);
    let abs = if cand.is_absolute() { cand } else { root.join(cand) };
    let canon = abs.canonicalize().ok()?;
    if canon.starts_with(root) {
        Some(canon)
    } else {
        None
    }
}

/// Resolve a (possibly not-yet-existing) output path: its parent must exist and
/// stay under `root`.
fn resolve_new(root: &Path, p: &str) -> Option<PathBuf> {
    let cand = PathBuf::from(p);
    let abs = if cand.is_absolute() { cand } else { root.join(cand) };
    let parent = abs.parent()?;
    let file = abs.file_name()?;
    let canon_parent = parent.canonicalize().ok()?;
    if canon_parent.starts_with(root) {
        Some(canon_parent.join(file))
    } else {
        None
    }
}

/// Drop leading separators, `.` and `..` components from an archive member name.
fn sanitize_rel(name: &str) -> PathBuf {
    let mut out = PathBuf::new();
    for comp in Path::new(name).components() {
        use std::path::Component::*;
        match comp {
            Normal(c) => out.push(c),
            _ => {} // ignore RootDir / ParentDir / CurDir / Prefix
        }
    }
    if out.as_os_str().is_empty() {
        out.push("file");
    }
    out
}

// ---------------------------------------------------------------------------
// HTTP / encoding helpers
// ---------------------------------------------------------------------------

fn split_query(url: &str) -> (&str, &str) {
    match url.split_once('?') {
        Some((p, q)) => (p, q),
        None => (url, ""),
    }
}

/// Extract and URL-decode a single field from a `&`-separated key=value string.
fn query_param(query: &str, key: &str) -> Option<String> {
    for pair in query.split('&') {
        if let Some((k, v)) = pair.split_once('=') {
            if k == key {
                return Some(url_decode(v));
            }
        }
    }
    None
}

fn url_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'+' => {
                out.push(b' ');
                i += 1;
            }
            b'%' if i + 2 < bytes.len() => {
                let hi = hex_val(bytes[i + 1]);
                let lo = hex_val(bytes[i + 2]);
                match (hi, lo) {
                    (Some(h), Some(l)) => {
                        out.push((h << 4) | l);
                        i += 3;
                    }
                    _ => {
                        out.push(bytes[i]);
                        i += 1;
                    }
                }
            }
            b => {
                out.push(b);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).to_string()
}

fn hex_val(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

/// Encode a string as a JSON string literal (with surrounding quotes).
fn json_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn err_json(msg: &str) -> String {
    format!("{{\"ok\":false,\"error\":{}}}", json_str(msg))
}

fn respond_json(request: tiny_http::Request, json: &str) -> Result<()> {
    let header = Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap();
    request
        .respond(Response::from_string(json).with_header(header))
        .map_err(|e| anyhow!("{e}"))
}

fn respond_html(request: tiny_http::Request, html: &str) -> Result<()> {
    let header =
        Header::from_bytes(&b"Content-Type"[..], &b"text/html; charset=utf-8"[..]).unwrap();
    request
        .respond(Response::from_string(html).with_header(header))
        .map_err(|e| anyhow!("{e}"))
}

const INDEX_HTML: &str = include_str!("gui/index.html");
