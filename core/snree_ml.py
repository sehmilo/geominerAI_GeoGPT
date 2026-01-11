import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List

from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)

EPS = 1e-9


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build consistent features for interval-level modeling.

    Expected (if available):
    - SnO2, Ta2O5, ZrO2, Rb2O, Cs2O, SrO, FracProxy
    - depth_mid_m, INTERVAL
    - hole_id, lat, lon
    """
    out = df.copy()
    numeric_cols = ["SnO2", "Ta2O5", "ZrO2", "Rb2O", "Cs2O", "SrO", "FracProxy", "depth_mid_m"]
    for c in numeric_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def build_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Backwards-compatible alias (some app versions import build_feature_table).
    """
    return build_features(df)


def _group_split(df: pd.DataFrame, test_size: float = 0.25, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Group split by hole_id to reduce leakage.
    """
    if "hole_id" in df.columns:
        groups = df["hole_id"].astype(str).fillna("NA")
    else:
        groups = df.index.astype(str)

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(df, groups=groups))
    return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()


def train_baselines(
    labeled: pd.DataFrame,
    target_col: str = "prospectivity_class",
    test_size: float = 0.25,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Trains robust baseline classifiers:
    - Logistic Regression (scaled, class_weight balanced)
    - RandomForest (balanced_subsample)

    Uses group split by hole_id when available.
    """
    data = labeled.dropna(subset=[target_col]).copy()

    feat_cols = [c for c in ["SnO2", "Ta2O5", "ZrO2", "FracProxy", "depth_mid_m"] if c in data.columns]
    if not feat_cols:
        raise ValueError("No feature columns found. Need at least one of: SnO2, Ta2O5, ZrO2, FracProxy, depth_mid_m")

    X = data[feat_cols].fillna(0.0)
    y = data[target_col].astype(str)

    merged = pd.concat([X, y, data.get("hole_id", pd.Series(index=data.index, dtype=str))], axis=1)
    train_df, test_df = _group_split(merged, test_size=test_size, seed=seed)

    X_train = train_df[feat_cols].values
    y_train = train_df[target_col].values
    X_test = test_df[feat_cols].values
    y_test = test_df[target_col].values

    classes_sorted = sorted(pd.unique(y).tolist())
    models: Dict[str, Any] = {}

    # Logistic Regression
    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))
    ])
    lr.fit(X_train, y_train)
    pred_lr = lr.predict(X_test)

    models["logreg"] = {
        "model": lr,
        "accuracy": float(accuracy_score(y_test, pred_lr)),
        "report": classification_report(y_test, pred_lr, output_dict=True, zero_division=0),
        "confusion": confusion_matrix(y_test, pred_lr, labels=classes_sorted),
        "features": feat_cols,
    }

    # RandomForest
    rf = RandomForestClassifier(
        n_estimators=400,
        random_state=seed,
        class_weight="balanced_subsample",
        min_samples_leaf=2,
    )
    rf.fit(X_train, y_train)
    pred_rf = rf.predict(X_test)

    models["random_forest"] = {
        "model": rf,
        "accuracy": float(accuracy_score(y_test, pred_rf)),
        "report": classification_report(y_test, pred_rf, output_dict=True, zero_division=0),
        "confusion": confusion_matrix(y_test, pred_rf, labels=classes_sorted),
        "features": feat_cols,
        "feature_importance": dict(zip(feat_cols, rf.feature_importances_.tolist())),
    }

    return {
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "classes": classes_sorted,
        "models": models,
    }


def within_hole_prediction(
    df: pd.DataFrame,
    hole_col: str = "hole_id",
    value_col: str = "SnO2",
) -> Dict[str, Any]:
    """
    Fit a simple depth trend per hole using interval midpoints, then predict a smooth curve.
    Returns both:
    - measured points (depth_mid_m, value, interval)
    - predicted curve (depth_grid_m, predicted)

    Important: This is a trend model for decision support, not a resource estimate.
    """
    out = df.copy()

    if hole_col not in out.columns:
        out[hole_col] = "HOLE"

    out[value_col] = pd.to_numeric(out.get(value_col, np.nan), errors="coerce")
    out["depth_mid_m"] = pd.to_numeric(out.get("depth_mid_m", np.nan), errors="coerce")

    # Unified total depth column
    out["total_depth_m"] = pd.to_numeric(
        out.get("total_depth_m", out.get("depth", out.get("pit_depth", out.get("total_depth", np.nan)))),
        errors="coerce"
    )

    # Interval label (optional, but helps plotting)
    if "INTERVAL" not in out.columns:
        out["INTERVAL"] = np.nan

    results: Dict[str, Any] = {}

    for hole, g in out.groupby(hole_col):
        gg = g.dropna(subset=[value_col, "depth_mid_m"]).copy()
        if len(gg) < 3:
            continue

        X = gg[["depth_mid_m"]].values
        y = gg[value_col].values

        reg = LinearRegression()
        reg.fit(X, y)

        td = gg["total_depth_m"].dropna()
        max_depth = float(td.max()) if len(td) else float(np.nanmax(gg["depth_mid_m"]))

        depth_grid = np.linspace(0.0, max_depth, 60).reshape(-1, 1)
        pred = reg.predict(depth_grid)

        results[str(hole)] = {
            "measured_depth_m": gg["depth_mid_m"].tolist(),
            "measured_value": gg[value_col].tolist(),
            "measured_interval": gg["INTERVAL"].astype(str).tolist(),
            "depth_grid_m": depth_grid.flatten().tolist(),
            "predicted": pred.tolist(),
            "n_points": int(len(gg)),
            "max_depth_m": float(max_depth),
        }

    return results
