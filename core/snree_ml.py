from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any, List

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import joblib

from core.snree_lab import SnREEConfig, _pick_col, _safe_numeric, _normalize_col


@dataclass
class FeatureCols:
    hole_id: Optional[str]
    depth: Optional[str]
    interval: Optional[str]
    lat: Optional[str]
    lon: Optional[str]

    sn: Optional[str]
    ta: Optional[str]
    zr: Optional[str]
    rb: Optional[str]
    cs: Optional[str]
    sr: Optional[str]


def detect_feature_cols(df: pd.DataFrame, cfg: SnREEConfig = SnREEConfig()) -> FeatureCols:
    # Core IDs
    hole_id = _pick_col(df, ("hole_id", "hole", "id", "sample_id", "sample"))
    depth = _pick_col(df, ("depth",))
    interval = _pick_col(df, ("interval", "INTERVAL"))
    lat = _pick_col(df, ("lat", "latitude"))
    lon = _pick_col(df, ("lon", "longitude", "long"))

    # Chemistry (oxide or element)
    sn = _pick_col(df, cfg.sn_candidates)
    ta = _pick_col(df, cfg.ta_candidates)
    zr = _pick_col(df, cfg.zr_candidates)
    rb = _pick_col(df, cfg.rb_candidates)
    cs = _pick_col(df, cfg.cs_candidates)
    sr = _pick_col(df, cfg.sr_candidates)

    return FeatureCols(hole_id, depth, interval, lat, lon, sn, ta, zr, rb, cs, sr)


def build_feature_table(df: pd.DataFrame, cfg: SnREEConfig = SnREEConfig()) -> Tuple[pd.DataFrame, FeatureCols]:
    """
    Row-level feature table for ML.
    Keeps lat/lon and depth if present.
    Creates fractionation proxy (Rb+Cs)/Sr.
    """
    cols = detect_feature_cols(df, cfg)
    d = df.copy()

    # Coerce numeric for chemical columns
    for c in [cols.sn, cols.ta, cols.zr, cols.rb, cols.cs, cols.sr, cols.depth, cols.lat, cols.lon]:
        if c and c in d.columns:
            d[c] = _safe_numeric(d[c])

    # Fractionation proxy
    if (cols.rb or cols.cs) and cols.sr:
        rb_vals = d[cols.rb].fillna(0) if cols.rb else 0.0
        cs_vals = d[cols.cs].fillna(0) if cols.cs else 0.0
        d["frac_proxy"] = (rb_vals + cs_vals) / (d[cols.sr].fillna(0) + 1e-9)
    else:
        d["frac_proxy"] = np.nan

    # Assemble feature frame
    out = pd.DataFrame()
    if cols.hole_id: out["hole_id"] = d[cols.hole_id].astype(str)
    if cols.interval: out["interval"] = d[cols.interval]
    if cols.depth: out["depth"] = d[cols.depth]
    if cols.lat: out["lat"] = d[cols.lat]
    if cols.lon: out["lon"] = d[cols.lon]

    # Chemical features
    out["sn"] = d[cols.sn] if cols.sn else np.nan
    out["ta"] = d[cols.ta] if cols.ta else np.nan
    out["zr"] = d[cols.zr] if cols.zr else np.nan
    out["rb"] = d[cols.rb] if cols.rb else np.nan
    out["cs"] = d[cols.cs] if cols.cs else np.nan
    out["sr"] = d[cols.sr] if cols.sr else np.nan
    out["frac_proxy"] = d["frac_proxy"]

    # Clean
    out = out.replace([np.inf, -np.inf], np.nan)

    return out, cols


def _label_row_rules(row: pd.Series, cfg: SnREEConfig) -> str:
    """
    Rule label per row (not p95). This makes a training target.
    Uses fold enrichment vs median computed later.
    """
    # Placeholder, replaced by fold-based labeling after medians computed.
    return "Low"


def label_by_rules(features: pd.DataFrame, cfg: SnREEConfig = SnREEConfig()) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Produces a labeled dataset using rule screening.
    Uses fold vs median computed on the same dataset.
    """
    f = features.copy()

    med = {
        "sn": float(np.nanmedian(f["sn"])) if f["sn"].notna().any() else 0.0,
        "ta": float(np.nanmedian(f["ta"])) if f["ta"].notna().any() else 0.0,
        "zr": float(np.nanmedian(f["zr"])) if f["zr"].notna().any() else 0.0,
        "frac_proxy": float(np.nanmedian(f["frac_proxy"])) if f["frac_proxy"].notna().any() else 0.0,
    }

    def fold(x: float, m: float) -> float:
        if m == 0.0:
            return float("inf") if (x and x > 0) else 0.0
        return float(x) / float(m)

    labels: List[str] = []
    sn_fold_list: List[float] = []
    ta_fold_list: List[float] = []
    zr_fold_list: List[float] = []
    frac_fold_list: List[float] = []

    for _, r in f.iterrows():
        snv = r.get("sn", np.nan)
        tav = r.get("ta", np.nan)
        zrv = r.get("zr", np.nan)
        frv = r.get("frac_proxy", np.nan)

        snf = fold(snv, med["sn"]) if pd.notna(snv) else 0.0
        taf = fold(tav, med["ta"]) if pd.notna(tav) else 0.0
        zrf = fold(zrv, med["zr"]) if pd.notna(zrv) else 0.0
        frf = fold(frv, med["frac_proxy"]) if pd.notna(frv) else 0.0

        sn_fold_list.append(snf)
        ta_fold_list.append(taf)
        zr_fold_list.append(zrf)
        frac_fold_list.append(frf)

        high = (
            (snf >= cfg.high_fold and taf >= cfg.high_fold)
            or (snf >= cfg.high_fold and frf >= cfg.high_fold)
            or (snf >= cfg.high_fold and zrf >= cfg.high_fold)
        )
        medium = (
            (snf >= cfg.medium_fold)
            or (taf >= cfg.medium_fold)
            or (frf >= cfg.medium_fold)
            or (zrf >= cfg.medium_fold)
        )

        if high:
            labels.append("High")
        elif medium:
            labels.append("Medium")
        else:
            labels.append("Low")

    f["sn_fold"] = sn_fold_list
    f["ta_fold"] = ta_fold_list
    f["zr_fold"] = zr_fold_list
    f["frac_fold"] = frac_fold_list
    f["prospectivity_class"] = labels

    return f, med


def train_logreg(labeled: pd.DataFrame, test_size: float = 0.25, seed: int = 42) -> Dict[str, Any]:
    """
    Train a baseline logistic regression model.
    Returns model + evaluation artifacts.
    """
    use_cols = ["sn", "ta", "zr", "rb", "cs", "sr", "frac_proxy", "depth"]
    # Keep only numeric columns that exist
    use_cols = [c for c in use_cols if c in labeled.columns]

    data = labeled.dropna(subset=["prospectivity_class"]).copy()
    X = data[use_cols].fillna(0.0)
    y = data["prospectivity_class"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y if y.nunique() > 1 else None
    )

    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("clf", LogisticRegression(max_iter=2000, multi_class="auto")),
    ])

    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    report = classification_report(y_test, preds, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, preds, labels=sorted(y.unique()))
    acc = accuracy_score(y_test, preds)

    return {
        "model": pipe,
        "features": use_cols,
        "accuracy": float(acc),
        "labels": sorted(y.unique()),
        "confusion_matrix": cm,
        "report": report,
        "test_rows": int(len(X_test)),
    }


def train_xgboost(labeled: pd.DataFrame, test_size: float = 0.25, seed: int = 42) -> Dict[str, Any]:
    """
    Optional: XGBoost baseline.
    Only works if xgboost is installed.
    """
    import xgboost as xgb

    use_cols = ["sn", "ta", "zr", "rb", "cs", "sr", "frac_proxy", "depth"]
    use_cols = [c for c in use_cols if c in labeled.columns]

    data = labeled.dropna(subset=["prospectivity_class"]).copy()
    X = data[use_cols].fillna(0.0)
    y_str = data["prospectivity_class"].astype(str)

    # Map labels to ints
    classes = sorted(y_str.unique())
    y = y_str.map({c: i for i, c in enumerate(classes)})

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y if y.nunique() > 1 else None
    )

    clf = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        num_class=len(classes),
        eval_metric="mlogloss",
        random_state=seed,
    )
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)

    acc = accuracy_score(y_test, pred)
    cm = confusion_matrix(y_test, pred, labels=list(range(len(classes))))

    return {
        "model": clf,
        "features": use_cols,
        "accuracy": float(acc),
        "labels": classes,
        "confusion_matrix": cm,
        "test_rows": int(len(X_test)),
    }


def save_model(bundle: Dict[str, Any], path: str) -> None:
    joblib.dump(bundle, path)


def load_model(path: str) -> Dict[str, Any]:
    return joblib.load(path)


def predict_ml(bundle: Dict[str, Any], features_row: pd.DataFrame) -> Tuple[str, Dict[str, float]]:
    """
    Predict class and probabilities (if available).
    """
    model = bundle["model"]
    cols = bundle["features"]
    labels = bundle["labels"]

    X = features_row[cols].fillna(0.0)

    pred = model.predict(X)[0]

    probs: Dict[str, float] = {}
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)[0]
        for i, lab in enumerate(labels):
            probs[lab] = float(p[i])

    return str(pred), probs
