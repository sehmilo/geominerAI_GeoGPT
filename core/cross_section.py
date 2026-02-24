"""
core/cross_section.py
Geological cross-section construction and schematic visualisation.

Combines:
  1. A professional structural-geology LLM prompt (user-specified spec)
  2. A matplotlib schematic figure built from available layer data
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# ── LLM prompt templates ──────────────────────────────────────────────────────

_SYSTEM = (
    "You are a professional structural geologist and mineral exploration expert. "
    "Your task is to construct and critically evaluate a geological cross section "
    "from the provided geological map data. "
    "You must follow strict geological principles and explicitly state all assumptions. "
    "CRITICAL INSTRUCTIONS: "
    "Do not fabricate data. "
    "Do not assume vertical beds unless supported. "
    "Do not change stratigraphic order without tectonic explanation. "
    "Clearly separate data-constrained interpretation from inferred interpretation. "
    "Prioritize geological plausibility over visual symmetry."
)

_TASK_TEMPLATE = """
INPUT DATA:
{input_data}

TASKS:
1. Establish Section Orientation
   - Determine orientation of section line relative to strike.
   - Calculate whether dip values will appear as true dip or apparent dip.
   - If apparent dip, calculate corrected true dip.

2. Construct Topographic Profile
   - Extract elevation along section line (estimate if DEM not provided).
   - Plot topographic profile before inserting geology.
   - State vertical exaggeration if used.

3. Project Surface Contacts to Section
   - Identify where section line intersects lithological contacts.
   - Mark each intersection on topographic profile.
   - Preserve stratigraphic order unless structural evidence suggests otherwise.

4. Project Units to Depth
   - Use dip angle and dip direction to project units downward.
   - Maintain consistent thickness unless deformation, folding, faulting,
     or intrusion justifies change.
   - Clearly distinguish inferred geometry from data-constrained geometry.

5. Interpret Structures
   - Identify folds (anticline, syncline, monocline).
   - Identify fault type (normal, reverse, thrust, strike-slip).
   - Apply correct displacement direction and offset magnitude.
   - Ensure no stratigraphic violations occur unless tectonically justified.

6. Handle Uncertainty
   - If data is sparse, propose the most geologically plausible model.
   - State all assumptions clearly.
   - Offer at least two alternative structural interpretations where ambiguity exists.

7. Validate Internal Consistency
   Check for: impossible unit stacking, inconsistent bed thickness,
   fault displacement errors, geometric impossibilities.
   Flag and explain any inconsistencies.

8. Mineral Exploration Relevance
   - Identify possible structural traps or mineralization controls.
   - Highlight zones of structural intersection.
   - Suggest where drilling would best test the model.

OUTPUT REQUIREMENTS:
Provide:
  a) Clear written explanation of reasoning
  b) Step-by-step structural logic
  c) Summary of assumptions (labelled clearly)
  d) List of uncertainties
  e) Exploration drilling recommendations
  f) A schematic ASCII cross-section or coordinate description
"""


# ── public API ────────────────────────────────────────────────────────────────

def build_cross_section_prompt(user_input: str, layer_context: str) -> str:
    """Return the full LLM prompt for cross-section construction."""
    input_data = (
        f"User description / command:\n{user_input}\n\n"
        f"Available data from loaded layers:\n{layer_context}"
    )
    return _SYSTEM + "\n\n" + _TASK_TEMPLATE.format(input_data=input_data)


def parse_units_from_text(text: str) -> List[Dict]:
    """
    Heuristically extract geological unit names and dip data from free text.
    Returns [{name, dip, dip_dir, color}, ...].
    """
    _COLORS = [
        "#c2956e", "#8fbc8f", "#6495ed", "#d4a0a0",
        "#b8d4b8", "#a0a0d4", "#d4c8a0", "#c8a0d4",
        "#f0c080", "#90d0c0",
    ]

    # Regex: "granite dips 30° NE" or "sandstone: strike 045, dip 25"
    hits = re.findall(
        r"(\w+(?:\s+\w+)?)\s+(?:dips?|strike[s]?)\s+(\d+)[°\s]*(N|S|E|W|NE|NW|SE|SW)?",
        text, re.IGNORECASE,
    )

    units = []
    seen  = set()
    for i, (name, dip, direction) in enumerate(hits):
        key = name.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        units.append({
            "name":    name.strip().title(),
            "dip":     min(int(dip), 89),
            "dip_dir": (direction or "NE").upper(),
            "color":   _COLORS[len(units) % len(_COLORS)],
        })

    # Fallback if nothing parsed
    if not units:
        defaults = [
            ("Upper Unit",  10, "NE"),
            ("Middle Unit", 25, "NE"),
            ("Lower Unit",  40, "NE"),
        ]
        for i, (nm, dp, dd) in enumerate(defaults):
            units.append({"name": nm, "dip": dp, "dip_dir": dd,
                          "color": _COLORS[i]})
    return units


def parse_faults_from_text(text: str) -> List[Dict]:
    """Extract fault references from user text."""
    faults   = []
    patterns = [
        (r"normal\s+fault",    "normal"),
        (r"reverse\s+fault",   "reverse"),
        (r"thrust",            "thrust"),
        (r"strike.slip",       "strike-slip"),
        (r"\bfault\b",         "unknown"),
    ]
    for pat, ftype in patterns:
        if re.search(pat, text, re.IGNORECASE):
            faults.append({"type": ftype})
            break
    return faults


def build_layer_context(layers: list) -> str:
    """Summarise available layers as text for the LLM prompt."""
    if not layers:
        return "No data layers loaded yet."

    lines = []
    for lyr in layers:
        t = lyr.get("type", "unknown")
        n = lyr.get("name", "unnamed")
        if t == "csv" and lyr.get("dataframe") is not None:
            df   = lyr["dataframe"]
            cols = list(df.columns)
            lines.append(f"  [CSV] {n} — columns: {cols}, rows: {len(df)}")
        elif t == "geojson" and lyr.get("geodata"):
            fc = lyr["geodata"]
            lines.append(f"  [GeoJSON] {n} — features: {len(fc.get('features', []))}")
        elif t == "raster":
            meta = lyr.get("metadata", {})
            lines.append(f"  [Raster] {n} — {meta.get('note', meta)}")
        elif t in ("pdf", "word", "text"):
            lines.append(f"  [{t.upper()}] {n} — {len(lyr.get('chunks', []))} text chunks")
        else:
            lines.append(f"  [{t.upper()}] {n}")
    return "\n".join(lines)


def generate_cross_section_figure(
    units:            List[Dict],
    section_length_m: float = 1000.0,
    topo_profile:     Optional[List[float]] = None,
    section_label:    str = "A — A′",
    faults:           Optional[List[Dict]] = None,
    strike_dips:      Optional[List[Dict]] = None,  # [{x_m, dip, label}]
) -> plt.Figure:
    """
    Generate a schematic geological cross-section figure.

    Parameters
    ----------
    units            : list of {name, dip, dip_dir, color}
    section_length_m : total section length in metres
    topo_profile     : optional elevation list (interpolated to 100 points)
    section_label    : e.g. "A — A′"
    faults           : optional list of {x_m, type}
    strike_dips      : optional list of {x_m, dip, label} for tick symbols
    """
    N = 120
    x = np.linspace(0, section_length_m, N)

    # ── topographic profile ───────────────────────────────────────────────────
    if topo_profile and len(topo_profile) >= 2:
        topo = np.interp(x,
                         np.linspace(0, section_length_m, len(topo_profile)),
                         topo_profile)
    else:
        # Synthetic gentle ridge
        topo = (120
                + 40 * np.sin(np.pi * x / section_length_m)
                + 8  * np.sin(3 * np.pi * x / section_length_m))

    topo_max = float(topo.max())
    topo_min = float(topo.min())

    n_units       = max(len(units), 1)
    section_depth = (topo_max - topo_min + 50) * 1.8
    unit_thick    = section_depth / n_units

    # ── figure setup ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(15, 7))
    fig.patch.set_facecolor("#f8f6f0")
    ax.set_facecolor("#f8f6f0")

    # Sky / above topography
    ax.fill_between(x, topo, topo_max + 40,
                    color="#d6eaf8", alpha=0.6, zorder=1)

    # ── geological units (dipping bands) ─────────────────────────────────────
    legend_patches = []
    for i, unit in enumerate(units):
        dip_rad  = np.radians(unit["dip"])
        sign     = 1.0 if unit["dip_dir"] in ("NE", "E", "SE", "N") else -1.0
        # Ramp: linear dip-tilt across section
        ramp     = sign * (x - section_length_m * 0.5) * np.tan(dip_rad) * 0.18

        top    = topo - i * unit_thick + ramp
        bottom = topo - (i + 1) * unit_thick + ramp

        ax.fill_between(x, bottom, top,
                        color=unit["color"], alpha=0.80, zorder=2 + i,
                        linewidth=0)
        ax.plot(x, top, color=unit["color"], linewidth=0.6,
                alpha=0.9, zorder=3 + i)

        # Bedding / foliation tick lines inside unit
        for pct in (0.2, 0.5, 0.8):
            xi = section_length_m * pct
            yi = float(np.interp(xi, x, (top + bottom) / 2))
            ax.annotate("", xy=(xi + 18, yi - 5 * sign),
                        xytext=(xi - 18, yi + 5 * sign),
                        arrowprops=dict(arrowstyle="-", color=unit["color"],
                                        lw=1.2, alpha=0.6),
                        zorder=5)

        # Unit label
        mid_x = section_length_m * 0.5
        mid_y = float(np.interp(mid_x, x, (top + bottom) / 2))
        ax.text(mid_x, mid_y,
                f"{unit['name']}\n({unit['dip']}° {unit['dip_dir']})",
                ha="center", va="center", fontsize=8, fontweight="bold",
                color="black",
                bbox=dict(boxstyle="round,pad=0.25",
                          facecolor=unit["color"], alpha=0.75,
                          edgecolor="gray", linewidth=0.5),
                zorder=10)

        legend_patches.append(
            mpatches.Patch(color=unit["color"],
                           label=f"{unit['name']} ({unit['dip']}° {unit['dip_dir']})")
        )

    # ── topographic surface ───────────────────────────────────────────────────
    ax.plot(x, topo, "k-", linewidth=2.0, zorder=20, label="Topographic surface")
    ax.fill_between(x, topo - 1, topo + 2, color="#5d4037",
                    alpha=0.3, zorder=19, label="Soil / regolith")

    # ── faults ────────────────────────────────────────────────────────────────
    if faults:
        fault_xs = np.linspace(section_length_m * 0.3,
                                section_length_m * 0.7,
                                len(faults))
        for fault, fx in zip(faults, fault_xs):
            ftype  = fault.get("type", "unknown")
            style  = "--" if "thrust" in ftype or "reverse" in ftype else "-"
            y_top  = float(np.interp(fx, x, topo))
            y_bot  = y_top - section_depth * 0.9
            ax.plot([fx, fx + sign * 30], [y_top, y_bot],
                    color="#c0392b", linewidth=2.2,
                    linestyle=style, zorder=25)
            ax.text(fx + 10, (y_top + y_bot) * 0.55,
                    f"Fault\n({ftype})", color="#c0392b",
                    fontsize=7, va="center", zorder=26,
                    bbox=dict(boxstyle="round,pad=0.15",
                              facecolor="white", alpha=0.7, edgecolor="#c0392b"))

    # ── strike/dip symbols ────────────────────────────────────────────────────
    if strike_dips:
        for sd in strike_dips:
            sx  = sd.get("x_m", section_length_m * 0.5)
            sy  = float(np.interp(sx, x, topo))
            dip = sd.get("dip", 30)
            ax.annotate(f"{dip}°", xy=(sx, sy + 8),
                        ha="center", fontsize=7, color="#1a5276",
                        fontweight="bold", zorder=30)

    # ── section end labels ────────────────────────────────────────────────────
    parts = [p.strip() for p in section_label.replace("—", "–").split("–")]
    lbl_a = parts[0] if parts else "A"
    lbl_b = parts[1] if len(parts) > 1 else "A′"
    ax.text(0, topo[0] + 28, lbl_a, fontsize=14, fontweight="bold",
            ha="center", va="bottom", color="#1a1a1a", zorder=35)
    ax.text(section_length_m, topo[-1] + 28, lbl_b, fontsize=14,
            fontweight="bold", ha="center", va="bottom",
            color="#1a1a1a", zorder=35)

    # ── uncertainty ribbon at base ────────────────────────────────────────────
    base_y = topo_min - section_depth * 0.92
    ax.axhline(base_y, color="#aaa", linewidth=0.8,
               linestyle=":", zorder=5, label="Inferred base (uncertain)")
    ax.fill_between(x, base_y - unit_thick * 0.3, base_y,
                    color="#bbb", alpha=0.2, zorder=4,
                    hatch="//", label="Zone of uncertainty")

    # ── scale bar ────────────────────────────────────────────────────────────
    bar_len   = section_length_m * 0.1
    bar_x     = section_length_m * 0.05
    bar_y     = topo_min - section_depth + unit_thick * 0.5
    ax.annotate("", xy=(bar_x + bar_len, bar_y),
                xytext=(bar_x, bar_y),
                arrowprops=dict(arrowstyle="<->", color="black", lw=1.5))
    ax.text(bar_x + bar_len / 2, bar_y - unit_thick * 0.25,
            f"{bar_len:.0f} m", ha="center", fontsize=8, color="black")

    # ── axes & legend ─────────────────────────────────────────────────────────
    ax.set_xlabel(f"Distance along section line (m)", fontsize=11)
    ax.set_ylabel("Elevation (m a.s.l.)", fontsize=11)
    ax.set_title(
        f"Geological Cross-Section  {section_label}  (Schematic — see interpretation below)",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.set_xlim(-section_length_m * 0.02, section_length_m * 1.02)
    ax.set_ylim(topo_min - section_depth, topo_max + 60)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.25, linestyle="--")

    legend_patches += [
        mpatches.Patch(color="#5d4037", alpha=0.4, label="Soil / regolith"),
        mpatches.Patch(facecolor="white", edgecolor="#c0392b",
                       label="Fault (red)"),
        mpatches.Patch(color="#bbb", alpha=0.4, hatch="//",
                       label="Zone of uncertainty"),
    ]
    ax.legend(handles=legend_patches,
              loc="lower right", fontsize=7.5, ncol=2,
              framealpha=0.85, edgecolor="#ccc")

    ax.text(0.01, 0.02,
            "⚠ Schematic only — based on available data and stated assumptions. "
            "Inferred portions shown with dashed lower boundary.",
            transform=ax.transAxes, fontsize=7.5, color="#666",
            style="italic", va="bottom")

    plt.tight_layout()
    return fig
