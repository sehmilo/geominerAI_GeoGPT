from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd


@dataclass
class SnREEConfig:
    # Your dataset uses oxide names, so we include both elemental and oxide variants.
    sample_id_candidates: Tuple[str, ...] = ("hole_id", "sample_id", "sampleid", "sample", "id", "name")

    sn_candidates: Tuple[str, ...] = ("sn", "sno2", "sn o2", "sn_ox", "sno")
    ta_candidates: Tuple[str, ...] = ("ta", "ta2o5", "ta2 o5", "ta_ox")
    nb_candidates: Tuple[str, ...] = ("nb", "nb2o5", "niobium")  # may not exist in your CSV
    zr_candidates: Tuple[str, ...] = ("zr", "zro2", "zr o2")
    rb_candidates: Tuple[str, ...] = ("rb", "rb2o", "rb o")
    cs_candidates: Tuple[str, ...] = ("cs", "cs2o", "cs o")
    sr_candidates: Tuple[str, ...] = ("sr", "sro", "sr o")

    # Ratio-based thresholds (unit-agnostic screening).
    # These do NOT assume ppm or wt%. They compare enrichment relative to dataset median.
    high_fold: float = 4.0
    medium_fold: float = 2.0


def _normalize_col(c: str) -> str:
    return str(c).strip().lower().replace("_", " ").replace("-", " ")


def _pick_col(df: pd.DataFrame, candidates: Tuple[str, ...]) -> Optional[str]:
    norm_map = {_normalize_col(c): c for c in df.columns}
    for cand in candidates:
        cand_norm = _normalize_col(cand)
        # exact match
        if cand_norm in norm_map:
            return norm_map[cand_norm]
        # partial contains match (e.g. "SnO2 (%)")
        for ncol, orig in norm_map.items():
            if cand_norm in ncol:
                return orig
    return None


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def load_xrf_csv(file) -> pd.DataFrame:
    return pd.read_csv(file)


def summarize_xrf(df: pd.DataFrame, cfg: SnREEConfig = SnREEConfig()) -> Dict[str, Any]:
    cols = {
        "sample_id": _pick_col(df, cfg.sample_id_candidates),
        "sn": _pick_col(df, cfg.sn_candidates),
        "ta": _pick_col(df, cfg.ta_candidates),
        "nb": _pick_col(df, cfg.nb_candidates),
        "zr": _pick_col(df, cfg.zr_candidates),
        "rb": _pick_col(df, cfg.rb_candidates),
        "cs": _pick_col(df, cfg.cs_candidates),
        "sr": _pick_col(df, cfg.sr_candidates),
    }

    df2 = df.copy()

    # Coerce detected columns to numeric
    for key in ("sn", "ta", "nb", "zr", "rb", "cs", "sr"):
        col = cols.get(key)
        if col:
            df2[col] = _safe_numeric(df2[col])

    # Fractionation proxy (very useful in Younger Granite context)
    # Higher (Rb+Cs)/Sr generally indicates more evolved/fractionated granite systems.
    rb_col, cs_col, sr_col = cols.get("rb"), cols.get("cs"), cols.get("sr")
    if (rb_col or cs_col) and sr_col:
        rb_vals = df2[rb_col] if rb_col else 0.0
        cs_vals = df2[cs_col] if cs_col else 0.0
        df2["_FRAC_PROXY_"] = (rb_vals.fillna(0) + cs_vals.fillna(0)) / (df2[sr_col].fillna(0) + 1e-9)
    else:
        df2["_FRAC_PROXY_"] = pd.NA

    def stats(colname: Optional[str]) -> Dict[str, Any]:
        if not colname or colname not in df2.columns:
            return {"count": 0}
        s = df2[colname].dropna()
        if len(s) == 0:
            return {"count": 0}
        return {
            "count": int(s.shape[0]),
            "max": float(s.max()),
            "p95": float(s.quantile(0.95)),
            "median": float(s.median()),
        }

    def top_rows(colname: Optional[str], n: int = 5) -> List[Dict[str, Any]]:
        if not colname or colname not in df2.columns:
            return []
        tmp = df2.dropna(subset=[colname]).sort_values(colname, ascending=False).head(n)
        out = []
        for _, r in tmp.iterrows():
            out.append({
                "sample": str(r.get(cols["sample_id"], "")) if cols["sample_id"] else "",
                "value": float(r[colname]),
            })
        return out

    summary = {
        "detected_columns": cols,
        "n_rows": int(len(df2)),
        "sn_stats": stats(cols["sn"]),
        "ta_stats": stats(cols["ta"]),
        "nb_stats": stats(cols["nb"]),
        "zr_stats": stats(cols["zr"]),
        "rb_stats": stats(cols["rb"]),
        "cs_stats": stats(cols["cs"]),
        "sr_stats": stats(cols["sr"]),
        "frac_proxy_stats": stats("_FRAC_PROXY_"),
        "top_sn": top_rows(cols["sn"], 5),
        "top_ta": top_rows(cols["ta"], 5),
        "top_zr": top_rows(cols["zr"], 5),
        "top_frac_proxy": top_rows("_FRAC_PROXY_", 5),
    }

    return summary


def quick_prospectivity(summary: Dict[str, Any], cfg: SnREEConfig = SnREEConfig()) -> Tuple[str, Dict[str, Any]]:
    """
    Unit-agnostic screening:
    - Uses fold-enrichment of p95 relative to median.
    - High if Sn and Ta are both strongly enriched, OR Sn strongly enriched + strong fractionation proxy.
    - Medium if one of Sn/Ta is moderately enriched or fractionation is elevated.
    """
    sn = summary.get("sn_stats", {})
    ta = summary.get("ta_stats", {})
    frac = summary.get("frac_proxy_stats", {})
    zr = summary.get("zr_stats", {})

    def fold(p95: float, median: float) -> float:
        if median is None or median == 0:
            return float("inf") if (p95 and p95 > 0) else 0.0
        return float(p95) / float(median)

    sn_fold = fold(sn.get("p95", 0.0) or 0.0, sn.get("median", 0.0) or 0.0)
    ta_fold = fold(ta.get("p95", 0.0) or 0.0, ta.get("median", 0.0) or 0.0)
    frac_fold = fold(frac.get("p95", 0.0) or 0.0, frac.get("median", 0.0) or 0.0)
    zr_fold = fold(zr.get("p95", 0.0) or 0.0, zr.get("median", 0.0) or 0.0)

    signals = {
        "Sn_p95": sn.get("p95", None),
        "Ta_p95": ta.get("p95", None),
        "FracProxy_p95": frac.get("p95", None),
        "Zr_p95": zr.get("p95", None),
        "Sn_fold_vs_median": sn_fold,
        "Ta_fold_vs_median": ta_fold,
        "Frac_fold_vs_median": frac_fold,
        "Zr_fold_vs_median": zr_fold,
    }

    high = (
        (sn_fold >= cfg.high_fold and ta_fold >= cfg.high_fold)
        or (sn_fold >= cfg.high_fold and frac_fold >= cfg.high_fold)
        or (sn_fold >= cfg.high_fold and zr_fold >= cfg.high_fold)
    )

    medium = (
        (sn_fold >= cfg.medium_fold)
        or (ta_fold >= cfg.medium_fold)
        or (frac_fold >= cfg.medium_fold)
        or (zr_fold >= cfg.medium_fold)
    )

    if high:
        return "High", signals
    if medium:
        return "Medium", signals
    return "Low", signals


def build_snree_prompt(
    question: str,
    xrf_summary: Dict[str, Any],
    evidence: List[Dict[str, Any]],
    prospectivity: str,
    signals: Dict[str, Any],
) -> str:
    lines: List[str] = []
    lines.append("You are GeoMinerAI Sn–REE Lab, an exploration geologist assistant.")
    lines.append("Project context: Jos Plateau Younger Granites (Sn–Nb–Ta province; REE potential may be inferred via fractionation proxies).")
    lines.append("Use ONLY the provided PDF evidence and the XRF summary below. Do not invent numbers, sample IDs, or locations.")
    lines.append("Cite PDF evidence in square brackets like [S1], [S2].")
    lines.append("")
    lines.append(f"Task: {question}")
    lines.append("")
    lines.append("XRF Summary (auto-extracted from CSV):")
    lines.append(f"- Rows: {xrf_summary.get('n_rows')}")
    lines.append(f"- Detected columns: {xrf_summary.get('detected_columns')}")
    lines.append(f"- Screening prospectivity: {prospectivity}")
    lines.append(f"- Signals: {signals}")
    lines.append(f"- Top SnO2/Sn samples: {xrf_summary.get('top_sn')}")
    lines.append(f"- Top Ta2O5/Ta samples: {xrf_summary.get('top_ta')}")
    lines.append(f"- Top ZrO2/Zr samples: {xrf_summary.get('top_zr')}")
    lines.append(f"- Top fractionation proxy samples: {xrf_summary.get('top_frac_proxy')}")
    lines.append("")
    lines.append("PDF Evidence:")
    for i, ev in enumerate(evidence, 1):
        lines.append(f"[S{i}] {ev['source']} (page {ev['page']}): {ev['text']}")
    lines.append("")
    lines.append("Output format (use these headings):")
    lines.append("1) Prospectivity Rating (High/Medium/Low)")
    lines.append("2) Exploration Vector (bullets only)")
    lines.append("3) Key XRF Evidence (bullets with numbers and hole_id)")
    lines.append("4) Interpretation for Jos Plateau Younger Granites (short paragraph)")
    lines.append("5) Uncertainties and Next Data to Collect (bullets)")
    lines.append("")
    lines.append("Write now.")
    return "\n".join(lines)
