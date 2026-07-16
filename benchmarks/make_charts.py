#!/usr/bin/env python3
"""Render the enwik8 benchmark charts for the README from results.csv.

Charts (PNG, light surface — GitHub READMEs render on white):
  1. benchmarks/enwik8_sizes.png    — compressed size vs well-known tools
  2. benchmarks/enwik8_tradeoff.png — size vs compress time across all levels
"""
import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = HERE
os.makedirs(OUT, exist_ok=True)

# --- palette (validated reference palette, light mode) ---------------------
BLUE = "#2a78d6"       # series 1: CPGC
GREEN = "#008300"      # series 2 (unused here; identity = CPGC vs others)
MUTED = "#898781"      # axis/labels
SECONDARY = "#52514e"
PRIMARY = "#0b0b0b"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"
NEUTRAL_BAR = "#c3c2b7"   # non-CPGC tools: recessive neutral
RESEARCH_BAR = "#9ec5f4"  # blue-100/200: research CM family (same hue, lighter)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "text.color": PRIMARY,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": SECONDARY,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})

# --- data -------------------------------------------------------------------
rows = {}
with open(os.path.join(HERE, "results.csv")) as f:
    for r in csv.DictReader(f):
        rows[r["mode"]] = r

levels = []
for lv in range(1, 10):
    key = f"cpgc-{lv}"
    if key not in rows:
        print(f"missing {key} in results.csv", file=sys.stderr)
        sys.exit(1)
    r = rows[key]
    assert r["verified"] == "1", f"{key} failed round-trip verification!"
    levels.append(
        (lv, int(r["comp_bytes"]), float(r["comp_seconds"]), float(r["decomp_seconds"]))
    )

# Headline: level 9 (max). Levels 8 and 9 produce identical archives today.
best = levels[8]

# LTCB published enwik8 figures (mattmahoney.net/dc/text.html) for context.
LTCB = [
    ("gzip -9", 36_445_248),
    ("zstd -22", 25_405_601),
    ("xz -9e", 24_703_772),
    ("bzip2 -9", 29_008_736),
    ("brotli -q11", 25_764_698),
    ("7z PPMd", 21_197_559),
    ("zpaq -m5", 17_855_729),
    ("paq8px", 16_046_995),
    ("cmix v21", 14_623_723),
]
V8_BASELINE = ("CPGC v8 (old)", 21_008_701)

# ---------------------------------------------------------------------------
# Chart 1: compressed size, CPGC v9 vs the field (horizontal bars, sorted)
# ---------------------------------------------------------------------------
entries = [("CPGC v9 -9 (this repo)", best[1], "cpgc")]
entries.append((V8_BASELINE[0], V8_BASELINE[1], "old"))
for name, size in LTCB:
    kind = "research" if name in ("zpaq -m5", "paq8px", "cmix v21") else "classic"
    entries.append((name, size, kind))
entries.sort(key=lambda e: e[1], reverse=True)

fig, ax = plt.subplots(figsize=(8.6, 5.4), dpi=160)
names = [e[0] for e in entries]
sizes = [e[1] / 1e6 for e in entries]
colors = {
    "cpgc": BLUE,
    "old": "#86b6ef",
    "classic": NEUTRAL_BAR,
    "research": "#e1e0d9",
}
bar_colors = [colors[e[2]] for e in entries]
bars = ax.barh(range(len(entries)), sizes, height=0.62, color=bar_colors, zorder=3)
ax.set_yticks(range(len(entries)), names)
for i, e in enumerate(entries):
    weight = "bold" if e[2] == "cpgc" else "normal"
    ink = PRIMARY if e[2] in ("cpgc", "old") else SECONDARY
    ax.text(
        sizes[i] + 0.35, i, f"{e[1]:,}",
        va="center", fontsize=8.5, color=ink, fontweight=weight,
    )
    ax.get_yticklabels()[i].set_fontweight(weight)
    ax.get_yticklabels()[i].set_color(ink)
ax.invert_yaxis()
ax.set_xlim(0, 42)
ax.set_xlabel("compressed size of enwik8 (MB) — smaller is better", fontsize=9)
ax.xaxis.set_major_locator(MultipleLocator(5))
ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
ax.spines[["top", "right", "left"]].set_visible(False)
ax.tick_params(axis="y", length=0, labelsize=9)
ax.tick_params(axis="x", labelsize=8)
ax.set_title(
    "enwik8 (100 MB English Wikipedia) — compressed size",
    fontsize=11, fontweight="bold", loc="left", pad=14, color=PRIMARY,
)
ax.text(
    0, 1.015, "Large Text Compression Benchmark file · reference sizes from mattmahoney.net/dc/text.html",
    transform=ax.transAxes, fontsize=7.5, color=MUTED,
)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "enwik8_sizes.png"))
print("wrote enwik8_sizes.png")

# ---------------------------------------------------------------------------
# Chart 2: ratio/speed trade-off across all 9 levels (scatter + path)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.6, 5.2), dpi=160)

xs = [t[2] / 60 for t in levels]  # minutes
ys = [t[1] / 1e6 for t in levels]
ax.plot(xs, ys, color=BLUE, linewidth=2, zorder=3)
ax.scatter(xs, ys, s=42, color=BLUE, zorder=4, edgecolors=SURFACE, linewidths=1.5)
# Per-level label offsets, tuned so clustered points don't collide.
OFFSETS = {
    1: (10, 4, "left", "bottom"),
    2: (-10, 0, "right", "center"),
    3: (10, -4, "left", "top"),
    4: (0, -12, "center", "top"),
    5: (0, 8, "center", "bottom"),
    6: (4, -12, "center", "top"),
    7: (0, 8, "center", "bottom"),
    8: (0, 8, "center", "bottom"),
    9: (0, -12, "center", "top"),
}
for (lv, size, ct, dt), x, y in zip(levels, xs, ys):
    dx, dy, ha, va = OFFSETS[lv]
    ax.annotate(
        f"-{lv}", (x, y), textcoords="offset points",
        xytext=(dx, dy), ha=ha, va=va,
        fontsize=8.5, color=PRIMARY, fontweight="bold",
    )

# Locally measured classical tools (same machine, same file).
local_pts = []
for name in ("gzip-9", "bzip2-9", "xz-9e"):
    if name in rows:
        r = rows[name]
        local_pts.append((name, int(r["comp_bytes"]) / 1e6, float(r["comp_seconds"]) / 60))
for name, size_mb, t_min in local_pts:
    ax.scatter([t_min], [size_mb], s=42, color=NEUTRAL_BAR, zorder=4,
               edgecolors=SURFACE, linewidths=1.5)
    ax.annotate(
        name, (t_min, size_mb), textcoords="offset points", xytext=(0, 8),
        ha="center", fontsize=8.5, color=SECONDARY,
    )

ax.set_xlabel("compress time on enwik8, minutes (4-core container)", fontsize=9)
ax.set_ylabel("compressed size (MB)", fontsize=9)
ax.grid(color=GRID, linewidth=0.8, zorder=0)
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(labelsize=8)
ax.set_title(
    "CPGC v9 levels 1-9 on enwik8 — size vs compress time",
    fontsize=11, fontweight="bold", loc="left", pad=14, color=PRIMARY,
)
ax.text(
    0, 1.015, "every point round-trip verified (decompressed and CRC-checked) · classical tools measured on the same machine",
    transform=ax.transAxes, fontsize=7.5, color=MUTED,
)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "enwik8_tradeoff.png"))
print("wrote enwik8_tradeoff.png")

# Markdown table for the README
print()
print("| level | compressed (bytes) | bpb | compress | decompress | verified |")
print("|--:|--:|--:|--:|--:|:--|")
for lv, size, ct, dt in levels:
    print(
        f"| {lv} | {size:,} | {size * 8 / 1e8:.3f} | {ct:.0f} s | {dt:.0f} s | ✓ |"
    )

# ---------------------------------------------------------------------------
# Chart 3 (optional): enwik9 — rendered only when enwik9_results.csv exists
# ---------------------------------------------------------------------------
e9_csv = os.path.join(HERE, "enwik9_results.csv")
if os.path.exists(e9_csv):
    e9 = {}
    with open(e9_csv) as f:
        for r in csv.DictReader(f):
            e9[r["mode"]] = r

    LTCB9 = [
        ("gzip -9", 322_591_995, "classic"),
        ("bzip2 -9", 253_977_839, "classic"),
        ("brotli -q11", 223_597_884, "classic"),
        ("zstd -22", 215_674_670, "classic"),
        ("xz -9e", 197_331_816, "classic"),
        ("7z PPMd", 178_965_454, "classic"),
        ("zpaq -m5", 142_252_605, "research"),
        ("paq8px", 126_486_867, "research"),
        ("cmix v21", 107_963_380, "research"),
        ("nncp v3.2", 106_632_363, "research"),
    ]

    best9 = e9["cpgc-9"]
    assert best9["verified"] == "1", "enwik9 -9 failed round-trip verification!"
    entries9 = [("CPGC v9 -9 (this repo)", int(best9["comp_bytes"]), "cpgc")]
    entries9 += LTCB9
    entries9.sort(key=lambda e: e[1], reverse=True)

    fig, ax = plt.subplots(figsize=(8.6, 5.6), dpi=160)
    names = [e[0] for e in entries9]
    sizes = [e[1] / 1e6 for e in entries9]
    bar_colors = [colors[e[2]] for e in entries9]
    ax.barh(range(len(entries9)), sizes, height=0.62, color=bar_colors, zorder=3)
    ax.set_yticks(range(len(entries9)), names)
    for i, e in enumerate(entries9):
        weight = "bold" if e[2] == "cpgc" else "normal"
        ink = PRIMARY if e[2] == "cpgc" else SECONDARY
        ax.text(sizes[i] + 3, i, f"{e[1]:,}",
                va="center", fontsize=8.5, color=ink, fontweight=weight)
        ax.get_yticklabels()[i].set_fontweight(weight)
        ax.get_yticklabels()[i].set_color(ink)
    ax.invert_yaxis()
    ax.set_xlim(0, 380)
    ax.set_xlabel("compressed size of enwik9 (MB) — smaller is better", fontsize=9)
    ax.xaxis.set_major_locator(MultipleLocator(50))
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0, labelsize=9)
    ax.tick_params(axis="x", labelsize=8)
    ax.set_title(
        "enwik9 (1 GB English Wikipedia) — compressed size",
        fontsize=11, fontweight="bold", loc="left", pad=14, color=PRIMARY,
    )
    ax.text(
        0, 1.015,
        "the current LTCB / Hutter Prize file · reference sizes from mattmahoney.net/dc/text.html",
        transform=ax.transAxes, fontsize=7.5, color=MUTED,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "enwik9_sizes.png"))
    print("wrote enwik9_sizes.png")

    print()
    print("| level | compressed (bytes) | bpb | compress | decompress | verified |")
    print("|--:|--:|--:|--:|--:|:--|")
    for lv in (1, 3, 5, 9):
        key = f"cpgc-{lv}"
        if key in e9:
            r = e9[key]
            size, ct, dt = int(r["comp_bytes"]), float(r["comp_seconds"]), float(r["decomp_seconds"])
            mark = "✓" if r["verified"] == "1" else "FAILED"
            print(f"| {lv} | {size:,} | {size * 8 / 1e9:.3f} | {ct/60:.0f} min | {dt/60:.0f} min | {mark} |")
