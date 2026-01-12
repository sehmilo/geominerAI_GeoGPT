import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any, List

from core.chemistry import ensure_oxides, standardize_columns

EPS = 1e-6

DEPTH_BINS = {
    "A": (0.0, 0.3),
    "B": (0.3, 1.0),
    "C": (1.0, 2.0),
    "D": (2.0, None),  # None means to pit_total_depth
}

# Explicit FracProxy definition (documented)
# FracProxy = (Rb2O + Cs2O) / (SrO + EPS)
# Why: Rb and Cs increase with granite differentiation, Sr generally decreases due to feldspar fractionation
def _compute_fracproxy(df: pd.DataFrame) -> pd.Series:
    rb = pd.to_numeric(df.get("Rb2O", np.nan), errors="coerce")
    cs = pd.to_numeric(df.get("Cs2O", np.nan), errors="coerce")
    sr = pd.to_numeric(df.get("SrO", np.nan), errors="coerce")
    return (rb.fillna(0.0) + cs.fillna(0.0)) / (sr.fillna(0.0) + EPS)


def load_xrf_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    df = standardize_columns(df)
    return df


def add_depth_interval_fields(df: pd.DataFrame, pit_depth_col: str = "depth") -> pd.DataFrame:
    """
    Adds interval depth start/end/mid based on INTERVAL A/B/C/D definitions.
    D uses pit_total_depth (pit_depth_col) as its end.
    """
    out = df.copy()
    if "INTERVAL" not in out.columns:
        raise ValueError("Missing INTERVAL column (expected A/B/C/D).")

    if pit_depth_col not in out.columns:
        # Try common alternatives
        for alt in ["pit_depth", "total_depth", "pit_total_depth", "depth_total", "Depth", "DEPTH"]:
            if alt in out.columns:
                pit_depth_col = alt
                break


    # Ensure numeric total depth
    out[pit_depth_col] = pd.to_numeric(out.get(pit_depth_col, np.nan), errors="coerce")
    out["total_depth_m"] = out[pit_depth_col]


    starts = []
    ends = []
    mids = []
    for _, r in out.iterrows():
        iv = str(r["INTERVAL"]).strip().upper()
        if iv not in DEPTH_BINS:
            starts.append(np.nan)
            ends.append(np.nan)
            mids.append(np.nan)
            continue

        a, b = DEPTH_BINS[iv]
        if iv == "D":
            td = r.get("total_depth_m", np.nan)
            if pd.isna(td) or td <= 2.0:
                starts.append(np.nan)
                ends.append(np.nan)
                mids.append(np.nan)
            else:
                starts.append(2.0)
                ends.append(float(td))
                mids.append((2.0 + float(td)) / 2.0)
        else:
            starts.append(a)
            ends.append(b)
            mids.append((a + b) / 2.0)

    out["depth_start_m"] = starts
    out["depth_end_m"] = ends
    out["depth_mid_m"] = mids
    return out


def summarize_xrf(df: pd.DataFrame, oxide_mode: str = "auto") -> Dict[str, Any]:
    """
    oxide_mode:
      - 'auto': keep oxides if present, else create from elements where possible
      - 'prefer_oxides': same as auto but explicit
      - 'elements_only': do not convert, keep as-is
    """
    if oxide_mode in ["auto", "prefer_oxides"]:
        dd = ensure_oxides(df, prefer_existing_oxides=True, allow_element_to_oxide=True)
    elif oxide_mode == "elements_only":
        dd = standardize_columns(df)
    else:
        raise ValueError("oxide_mode must be one of: auto, prefer_oxides, elements_only")

    # Core signals
    for col in ["SnO2", "Ta2O5", "ZrO2", "Rb2O", "Cs2O", "SrO"]:
        if col in dd.columns:
            dd[col] = pd.to_numeric(dd[col], errors="coerce")

    dd["FracProxy"] = _compute_fracproxy(dd)

    def p95(x):
        x2 = x.dropna()
        return float(np.percentile(x2, 95)) if len(x2) else float("nan")

    def median(x):
        x2 = x.dropna()
        return float(np.median(x2)) if len(x2) else float("nan")

    summary = {
        "n_rows": int(len(dd)),
        "has_latlon": bool(("lat" in dd.columns) and ("lon" in dd.columns)),
        "available_columns": list(dd.columns),
        "SnO2_p95": p95(dd["SnO2"]) if "SnO2" in dd.columns else float("nan"),
        "Ta2O5_p95": p95(dd["Ta2O5"]) if "Ta2O5" in dd.columns else float("nan"),
        "ZrO2_p95": p95(dd["ZrO2"]) if "ZrO2" in dd.columns else float("nan"),
        "FracProxy_p95": p95(dd["FracProxy"]),
        "SnO2_median": median(dd["SnO2"]) if "SnO2" in dd.columns else float("nan"),
        "Ta2O5_median": median(dd["Ta2O5"]) if "Ta2O5" in dd.columns else float("nan"),
        "ZrO2_median": median(dd["ZrO2"]) if "ZrO2" in dd.columns else float("nan"),
        "FracProxy_median": median(dd["FracProxy"]),
        "FracProxy_definition": "FracProxy = (Rb2O + Cs2O) / (SrO + 1e-6)",
        "FracProxy_rationale": "Rb and Cs are incompatible and often increase with granite differentiation; Sr often decreases due to feldspar fractionation. High FracProxy suggests a more evolved granite system, relevant for Sn, Nb, Ta, and REE enrichment.",
    }
    return summary


def quick_prospectivity(summary: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """
    Transparent screening rules. Intended as early-stage targeting, not ground truth.
    """
    sn_p95 = summary.get("SnO2_p95", np.nan)
    ta_p95 = summary.get("Ta2O5_p95", np.nan)
    zr_p95 = summary.get("ZrO2_p95", np.nan)
    fp_p95 = summary.get("FracProxy_p95", np.nan)

    sn_med = summary.get("SnO2_median", np.nan)
    ta_med = summary.get("Ta2O5_median", np.nan)
    zr_med = summary.get("ZrO2_median", np.nan)
    fp_med = summary.get("FracProxy_median", np.nan)

    def fold(p, m):
        if pd.isna(p) or pd.isna(m) or m == 0:
            return float("nan")
        return float(p / m)

    signals = {
        "Sn_p95": sn_p95,
        "Ta_p95": ta_p95,
        "Zr_p95": zr_p95,
        "FracProxy_p95": fp_p95,
        "Sn_fold_vs_median": fold(sn_p95, sn_med),
        "Ta_fold_vs_median": fold(ta_p95, ta_med),
        "Zr_fold_vs_median": fold(zr_p95, zr_med),
        "Frac_fold_vs_median": fold(fp_p95, fp_med),
        "FracProxy_definition": summary.get("FracProxy_definition"),
    }

    # Simple screening logic (tunable later)
    score = 0
    if not pd.isna(signals["Sn_fold_vs_median"]) and signals["Sn_fold_vs_median"] >= 5:
        score += 2
    if not pd.isna(signals["Ta_fold_vs_median"]) and signals["Ta_fold_vs_median"] >= 2:
        score += 1
    if not pd.isna(signals["Frac_fold_vs_median"]) and signals["Frac_fold_vs_median"] >= 1.5:
        score += 1
    if not pd.isna(signals["Zr_fold_vs_median"]) and signals["Zr_fold_vs_median"] >= 1.5:
        score += 1

    if score >= 4:
        rating = "High"
    elif score >= 2:
        rating = "Medium"
    else:
        rating = "Low"

    return rating, signals


def build_snree_prompt(question: str, xrf_summary: Dict[str, Any], evidence: List[Dict[str, Any]], prospectivity: str, signals: Dict[str, Any]) -> str:
    ev_txt = ""
    for i, h in enumerate(evidence, 1):
        ev_txt += f"\n[S{i}] {h['source']} page {h['page']}\n{h['text']}\n"

    prompt = f"""
You are a geologist specializing in granite-related Sn–Nb–Ta–REE systems (Jos Plateau Younger Granites context).
Task: {question}

INPUTS:
- Prospectivity screening (rules-only): {prospectivity}
- Signals: {signals}
- XRF Summary: {xrf_summary}

EVIDENCE (use citations like [S1], [S2] where relevant):
{ev_txt}

OUTPUT REQUIREMENTS:
1) Explain what the screening signals mean and why (tie to evolved granite, greisen, pegmatite, alteration concepts).
2) Provide an exploration vector: what to check next in the field, what minerals/alteration to look for, what follow-up samples to collect.
3) Keep it defensible. If evidence is missing, state assumptions clearly.
4) Cite evidence snippets when you use them.

Answer:
""".strip()

    return prompt
